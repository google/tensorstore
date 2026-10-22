"""HTTP/2 test server using asyncio for TensorStore transport testing.

Derived in part from the python-hyper/h2 asyncio-server example:
https://github.com/python-hyper/h2/blob/master/examples/asyncio/
asyncio-server.py

Extensively modified for TensorStore:
- Class-based Endpoint dispatching architecture.
- In-memory key-value store for PUT, POST, GET, HEAD, DELETE.
- Transparent compression testing (gzip Content-Encoding).
- Unusual frame sequences: a response missing :status, fragmented gzip with
  zero-length DATA and PINGs, and CONTINUATION-spanning headers.
- HTTP CONNECT proxy for tunneling tests.

Scripted stream resets, GOAWAY, flow control, interim responses, trailers and
stalls are covered by the in-process server in e2e_http2_test.cc; this
server provides TLS and an HTTP/2 implementation independent of nghttp2.

-----------------------------------------------------------------------------
The MIT License (MIT)

Copyright (c) 2015-2020 Cory Benfield and contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

-----------------------------------------------------------------------------

Copyright 2026 The TensorStore Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import abc
import argparse
import asyncio
import collections
import datetime
import gzip
import io
import json
import os
import ssl
import sys
from typing import Dict, List, Optional, Set, Tuple

from h2.config import H2Configuration
from h2.connection import H2Connection
from h2.errors import ErrorCodes
from h2.events import (
    ConnectionTerminated,
    DataReceived,
    RemoteSettingsChanged,
    RequestReceived,
    StreamEnded,
    StreamReset,
    WindowUpdated,
)
from h2.exceptions import ProtocolError, StreamClosedError
from h2.settings import SettingCodes


RequestData = collections.namedtuple('RequestData', ['headers', 'data'])


_DATA: Dict[str, RequestData] = {}
_PROXY_PEERS: Set[Tuple[str, int]] = set()


class Endpoint(abc.ABC):
  """Base class for HTTP/2 endpoint behaviors."""

  @abc.abstractmethod
  def matches(self, method: str, pure_path: str, req: RequestData) -> bool:
    """Return True if this endpoint handles the given request."""

  @abc.abstractmethod
  def handle(
      self, proto: 'H2Protocol', stream_id: int, req: RequestData
  ) -> None:
    """Handle the completed request stream."""


class MalformedNoStatusEndpoint(Endpoint):
  """Emits invalid headers missing mandatory :status pseudo-header."""

  def __init__(self, path: str = '/malformed/no_status'):
    self._path = path

  def matches(self, method: str, pure_path: str, req: RequestData) -> bool:
    return pure_path == self._path

  def handle(
      self, proto: 'H2Protocol', stream_id: int, req: RequestData
  ) -> None:
    response_headers = (
        ('content-type', 'text/plain'),
        ('x-custom', 'missing-status'),
    )
    proto.conn.send_headers(stream_id, response_headers, end_stream=True)
    proto.flush()


class FragmentedGzipEndpoint(Endpoint):
  """Streams gzip body in 2-byte DATA frames with 0-byte DATA and PINGs."""

  def __init__(self, path: str = '/fragmented_gzip'):
    self._path = path
    self._uncompressed = b'Fragmented gzip stream payload. ' * 8
    self._compressed = gzip.compress(self._uncompressed)

  def matches(self, method: str, pure_path: str, req: RequestData) -> bool:
    return pure_path == self._path

  def handle(
      self, proto: 'H2Protocol', stream_id: int, req: RequestData
  ) -> None:
    headers = (
        (':status', '200'),
        ('content-type', 'text/plain'),
        ('content-encoding', 'gzip'),
        ('content-length', str(len(self._compressed))),
        ('server', 'asyncio-h2'),
    )
    proto.conn.send_headers(stream_id, headers)
    # Leading 0-length DATA frame and server-initiated PING before any gzip
    # header bytes arrive.
    proto.conn.send_data(stream_id, b'', end_stream=False)
    proto.conn.ping(b'ping0001')
    proto.flush()

    mid = len(self._compressed) // 2
    for i in range(0, len(self._compressed), 2):
      if i == mid:
        proto.conn.send_data(stream_id, b'', end_stream=False)
        proto.conn.ping(b'ping0002')
      proto.conn.send_data(
          stream_id, self._compressed[i : i + 2], end_stream=False
      )
      proto.flush()

    # Trailing 0-length DATA frame carrying END_STREAM.
    proto.conn.send_data(stream_id, b'', end_stream=True)
    proto.flush()


class LargeHeadersEndpoint(Endpoint):
  """Emits >20 KiB of headers to force HEADERS + CONTINUATION frames."""

  def __init__(self, path: str = '/large_headers'):
    self._path = path
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    self._large_headers = []
    for i in range(8):
      val = ''.join(
          alphabet[(i * 13 + j * 7) % len(alphabet)] for j in range(3000)
      )
      self._large_headers.append((f'x-large-{i}', val))

  def matches(self, method: str, pure_path: str, req: RequestData) -> bool:
    return pure_path == self._path

  def handle(
      self, proto: 'H2Protocol', stream_id: int, req: RequestData
  ) -> None:
    body = b'continuation_headers_ok'
    headers = [
        (':status', '200'),
        ('content-type', 'text/plain'),
        ('content-length', str(len(body))),
        ('server', 'asyncio-h2'),
    ]
    headers.extend(self._large_headers)
    proto.conn.send_headers(stream_id, headers)
    proto.conn.send_data(stream_id, body, end_stream=True)
    proto.flush()


class CompressedResponseEndpoint(Endpoint):
  """Emits gzip-compressed body with Content-Encoding: gzip."""

  def __init__(self, path: str = '/compressed'):
    self._path = path
    pattern = (
        b'TensorStore HTTP/2 transparent compression test payload. '
        b'Repeatable string pattern for gzip compression efficiency. '
    )
    self._uncompressed_data = pattern * 10
    self._compressed_data = gzip.compress(self._uncompressed_data)

  def matches(self, method: str, pure_path: str, req: RequestData) -> bool:
    return pure_path == self._path

  def handle(
      self, proto: 'H2Protocol', stream_id: int, req: RequestData
  ) -> None:
    headers = (
        (':status', '200'),
        ('content-type', 'text/plain'),
        ('content-encoding', 'gzip'),
        ('content-length', str(len(self._compressed_data))),
        ('server', 'asyncio-h2'),
    )
    proto.conn.send_headers(stream_id, headers)
    proto.conn.send_data(stream_id, self._compressed_data, end_stream=True)
    proto.flush()


class RestStorageEndpoint(Endpoint):
  """REST CRUD storage handler for PUT, POST, GET, HEAD, DELETE."""

  def __init__(self, storage: Dict[str, RequestData]):
    self._storage = storage

  def matches(self, method: str, pure_path: str, req: RequestData) -> bool:
    return method in ('PUT', 'POST', 'GET', 'HEAD', 'DELETE')

  def handle(
      self, proto: 'H2Protocol', stream_id: int, req: RequestData
  ) -> None:
    method = req.headers[':method']
    path = req.headers[':path']
    if method in ('PUT', 'POST'):
      if 'etag' not in req.headers:
        req.headers['etag'] = str(datetime.datetime.now())
      self._storage[path] = req
      body = f'Received {len(req.data.getvalue())} bytes\n'.encode('utf-8')
      proto.send_response(stream_id, status='200', data=body)
    elif method == 'GET':
      if path not in self._storage:
        proto.send_404(stream_id, path)
        return
      proto.send_response(
          stream_id, status='200', data=self._storage[path].data.getvalue()
      )
    elif method == 'HEAD':
      if path not in self._storage:
        proto.send_404(stream_id, path)
        return
      headers = (
          (':status', '200'),
          ('content-type', 'text/plain'),
          ('content-length', str(len(self._storage[path].data.getvalue()))),
          ('server', 'asyncio-h2'),
      )
      proto.conn.send_headers(stream_id, headers)
      asyncio.ensure_future(proto.send_data(b'', stream_id))
    elif method == 'DELETE':
      if path not in self._storage:
        proto.send_404(stream_id, path)
        return
      del self._storage[path]
      proto.conn.send_headers(stream_id, [(':status', '200')])
      asyncio.ensure_future(proto.send_data(b'', stream_id))


class EchoEndpoint(Endpoint):
  """Fallback endpoint echoing request headers and body as JSON."""

  def matches(self, method: str, pure_path: str, req: RequestData) -> bool:
    return True

  def handle(
      self, proto: 'H2Protocol', stream_id: int, req: RequestData
  ) -> None:
    headers = req.headers
    body = req.data.getvalue().decode('utf-8')
    data = json.dumps({'headers': headers, 'body': body}, indent=4).encode(
        'utf-8'
    )
    headers_out = [
        (':status', '200'),
        ('content-type', 'application/json'),
        ('content-length', str(len(data))),
        ('server', 'asyncio-h2'),
    ]
    if proto.via_proxy:
      headers_out.append(('x-via-proxy', '1'))
    proto.conn.send_headers(stream_id, headers_out)
    asyncio.ensure_future(proto.send_data(data, stream_id))


def make_default_endpoints(storage: Dict[str, RequestData]) -> List[Endpoint]:
  """Creates default HTTP/2 test endpoints for regression and edge cases."""
  return [
      MalformedNoStatusEndpoint('/malformed/no_status'),
      FragmentedGzipEndpoint('/fragmented_gzip'),
      LargeHeadersEndpoint('/large_headers'),
      CompressedResponseEndpoint('/compressed'),
      RestStorageEndpoint(storage),
      EchoEndpoint(),
  ]


async def _pipe_stream(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
  """Forwards bytes from reader to writer until EOF or connection close."""
  try:
    while True:
      data = await reader.read(65536)
      if not data:
        break
      writer.write(data)
      await writer.drain()
  except (ConnectionError, asyncio.CancelledError):
    pass
  finally:
    try:
      writer.close()
    except Exception:  # pylint: disable=broad-exception-caught
      pass


async def handle_proxy_client(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
) -> None:
  """Minimal HTTP/1.1 CONNECT tunnel proxy for testing HTTP/2 over proxy."""
  backend_writer = None
  sockname = None
  try:
    header_data = await client_reader.readuntil(b'\r\n\r\n')
    req_line = header_data.split(b'\r\n', 1)[0].decode('latin1')
    parts = req_line.split(' ')
    if len(parts) < 2 or parts[0].upper() != 'CONNECT':
      client_writer.write(
          b'HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\n\r\n'
      )
      await client_writer.drain()
      return

    host_port = parts[1]
    host, port_str = host_port.rsplit(':', 1)
    if host.startswith('[') and host.endswith(']'):
      host = host[1:-1]
    port = int(port_str)

    backend_reader, backend_writer = await asyncio.open_connection(host, port)
    sockname = backend_writer.get_extra_info('sockname')
    if sockname is not None:
      _PROXY_PEERS.add(sockname)

    client_writer.write(b'HTTP/1.1 200 Connection Established\r\n\r\n')
    await client_writer.drain()

    await asyncio.gather(
        _pipe_stream(client_reader, backend_writer),
        _pipe_stream(backend_reader, client_writer),
    )
  except Exception:  # pylint: disable=broad-exception-caught
    try:
      client_writer.write(b'HTTP/1.1 502 Bad Gateway\r\n\r\n')
      await client_writer.drain()
    except Exception:  # pylint: disable=broad-exception-caught
      pass
  finally:
    if sockname is not None:
      _PROXY_PEERS.discard(sockname)
    if backend_writer is not None:
      backend_writer.close()
    client_writer.close()


class H2Protocol(asyncio.Protocol):

  def __init__(self, endpoints: Optional[List[Endpoint]] = None):
    config = H2Configuration(
        client_side=False,
        header_encoding='utf-8',
        validate_outbound_headers=False,
    )
    self.conn = H2Connection(config=config)
    self.transport = None
    self.stream_data = {}
    self.flow_control_futures = {}
    self.via_proxy = False
    self.endpoints = (
        endpoints if endpoints is not None else make_default_endpoints(_DATA)
    )

  def flush(self):
    """Flushes buffered protocol data to transport."""
    self.transport.write(self.conn.data_to_send())

  def send_response(
      self,
      stream_id: int,
      status: str = '200',
      data: bytes = b'',
      content_type: str = 'text/plain',
  ):
    """Sends status headers and body data."""
    headers = [
        (':status', str(status)),
        ('content-type', content_type),
        ('content-length', str(len(data))),
        ('server', 'asyncio-h2'),
    ]
    if self.via_proxy:
      headers.append(('x-via-proxy', '1'))
    self.conn.send_headers(stream_id, headers)
    asyncio.ensure_future(self.send_data(data, stream_id))

  def send_404(self, stream_id: int, path: str):
    """Sends standard 404 response."""
    msg = f'Not Found: {path}\n'.encode('utf-8')
    self.send_response(stream_id, status='404', data=msg)

  def connection_made(self, transport: asyncio.Transport):
    self.transport = transport
    peername = transport.get_extra_info('peername')
    self.via_proxy = peername in _PROXY_PEERS
    self.conn.initiate_connection()
    self.transport.write(self.conn.data_to_send())

  def connection_lost(self, exc):
    for future in self.flow_control_futures.values():
      future.cancel()
    self.flow_control_futures = {}

  def data_received(self, data: bytes):
    try:
      events = self.conn.receive_data(data)
    except ProtocolError:
      self.transport.write(self.conn.data_to_send())
      self.transport.close()
    else:
      self.transport.write(self.conn.data_to_send())
      for event in events:
        if isinstance(event, RequestReceived):
          self.request_received(event.headers, event.stream_id)
        elif isinstance(event, DataReceived):
          self.receive_data(
              event.data, event.flow_controlled_length, event.stream_id
          )
        elif isinstance(event, StreamEnded):
          self.stream_complete(event.stream_id)
        elif isinstance(event, ConnectionTerminated):
          self.transport.close()
        elif isinstance(event, StreamReset):
          self.stream_reset(event.stream_id)
        elif isinstance(event, WindowUpdated):
          self.window_updated(event.stream_id, event.delta)
        elif isinstance(event, RemoteSettingsChanged):
          if SettingCodes.INITIAL_WINDOW_SIZE in event.changed_settings:
            self.window_updated(None, 0)

        self.transport.write(self.conn.data_to_send())

  def request_received(self, headers: List[Tuple[str, str]], stream_id: int):
    headers = collections.OrderedDict(headers)

    # Store off the request data.
    request_data = RequestData(headers, io.BytesIO())
    self.stream_data[stream_id] = request_data

  def stream_complete(self, stream_id: int):
    """When a stream is complete, dispatch to matching endpoint."""
    try:
      request_data = self.stream_data[stream_id]
    except KeyError:
      return

    path = request_data.headers.get(':path', '')
    pure_path = path.split('?')[0]
    method = request_data.headers.get(':method', '')

    for endpoint in self.endpoints:
      if endpoint.matches(method, pure_path, request_data):
        endpoint.handle(self, stream_id, request_data)
        break

  def receive_data(
      self, data: bytes, flow_controlled_length: int, stream_id: int
  ):
    """We've received some data on a stream.

    If that stream is one we're expecting data on, save it off (and account for
    the received amount of data in flow control so that the client can send more
    data). Otherwise, reset the stream.
    """
    try:
      stream_data = self.stream_data[stream_id]
    except KeyError:
      self.conn.reset_stream(stream_id, error_code=ErrorCodes.PROTOCOL_ERROR)
    else:
      stream_data.data.write(data)
      self.conn.acknowledge_received_data(flow_controlled_length, stream_id)

  def stream_reset(self, stream_id):
    """A stream reset was sent. Stop sending data."""
    if stream_id in self.flow_control_futures:
      future = self.flow_control_futures.pop(stream_id)
      future.cancel()

  async def send_data(self, data, stream_id):
    """Send data according to the flow control rules."""
    if not data:
      try:
        self.conn.send_data(stream_id, b'', end_stream=True)
      except (StreamClosedError, ProtocolError):
        return
      self.transport.write(self.conn.data_to_send())
      return

    while data:
      while self.conn.local_flow_control_window(stream_id) < 1:
        try:
          await self.wait_for_flow_control(stream_id)
        except asyncio.CancelledError:
          return

      chunk_size = min(
          self.conn.local_flow_control_window(stream_id),
          len(data),
          self.conn.max_outbound_frame_size,
      )

      try:
        self.conn.send_data(
            stream_id, data[:chunk_size], end_stream=(chunk_size == len(data))
        )
      except (StreamClosedError, ProtocolError):
        # The stream got closed and we didn't get told. We're done
        # here.
        break

      self.transport.write(self.conn.data_to_send())
      data = data[chunk_size:]

  async def wait_for_flow_control(self, stream_id):
    """Waits for a Future that fires when the flow control window is opened."""
    f = asyncio.Future()
    self.flow_control_futures[stream_id] = f
    await f

  def window_updated(self, stream_id, delta):
    """A window update frame was received.

    Unblock some number of flow control Futures.
    """
    if stream_id and stream_id in self.flow_control_futures:
      f = self.flow_control_futures.pop(stream_id)
      f.set_result(delta)
    elif not stream_id:
      for f in self.flow_control_futures.values():
        f.set_result(delta)

      self.flow_control_futures = {}


def run(port, cert_path, proxy_port=0):
  """Starts the HTTP/2 TLS server and HTTP CONNECT proxy server."""
  if cert_path is None:
    cert_path = os.path.dirname(__file__)
  ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
  ssl_context.options |= ssl.OP_NO_COMPRESSION
  ssl_context.load_cert_chain(
      certfile=os.path.join(cert_path, 'test.crt'),
      keyfile=os.path.join(cert_path, 'test.key'),
  )
  ssl_context.set_alpn_protocols(['h2'])

  # Python 3.10+ requires a running event loop.
  if sys.version_info < (3, 10):
    loop = asyncio.get_event_loop()
  else:
    try:
      loop = asyncio.get_running_loop()
    except RuntimeError:
      loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

  # Each client connection will create a new protocol instance
  coro = loop.create_server(H2Protocol, '127.0.0.1', port, ssl=ssl_context)
  server = loop.run_until_complete(coro)
  proxy_coro = asyncio.start_server(
      handle_proxy_client, '127.0.0.1', proxy_port
  )
  proxy_server = loop.run_until_complete(proxy_coro)

  # Serve requests until Ctrl+C is pressed
  print('Serving proxy on {}'.format(proxy_server.sockets[0].getsockname()))
  print('Serving on {}'.format(server.sockets[0].getsockname()))
  sys.stdout.flush()

  try:
    loop.run_forever()
  except KeyboardInterrupt:
    pass
  finally:
    # Close the servers
    proxy_server.close()
    server.close()
    loop.run_until_complete(proxy_server.wait_closed())
    loop.run_until_complete(server.wait_closed())
    loop.close()


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--port', type=int, default=0)
  p.add_argument('--proxy_port', type=int, default=0)
  p.add_argument('--cert_path', type=str, default=None)
  v, _ = p.parse_known_args(sys.argv[1:])
  print(f'Starting h2_server with --port={v.port}')
  sys.exit(run(v.port, v.cert_path, v.proxy_port))
