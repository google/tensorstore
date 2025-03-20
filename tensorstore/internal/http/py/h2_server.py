"""Modified version of h2 asyncio-server.py

https://github.com/python-hyper/h2/blob/master/examples/asyncio/asyncio-server.py

A fully-functional HTTP/2 server using asyncio. Requires Python 3.5+.

Modified to store values in a dictionary, and to support PUT, DELETE, HEAD, and
GET requests.

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
"""

import argparse
import asyncio
import collections
import datetime
import io
import json
import os
import ssl
import sys
from typing import Dict, List, Tuple

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


class H1Protocol(asyncio.Protocol):

  def __init__(self):
    config = H2Configuration(client_side=False, header_encoding='utf-8')
    self.conn = H2Connection(config=config)
    self.transport = None
    self.stream_data = {}
    self.flow_control_futures = {}

  def connection_made(self, transport: asyncio.Transport):
    self.transport = transport
    self.transport.write(self.conn.data_to_send())

  def connection_lost(self, exc):
    for future in self.flow_control_futures.values():
      future.cancel()
    self.flow_control_futures = {}


class H2Protocol(asyncio.Protocol):

  def __init__(self):
    config = H2Configuration(client_side=False, header_encoding='utf-8')
    self.conn = H2Connection(config=config)
    self.transport = None
    self.stream_data = {}
    self.flow_control_futures = {}

  def connection_made(self, transport: asyncio.Transport):
    self.transport = transport
    self.conn.initiate_connection()
    self.transport.write(self.conn.data_to_send())

  def connection_lost(self, exc):
    for future in self.flow_control_futures.values():
      future.cancel()
    self.flow_control_futures = {}

  def data_received(self, data: bytes):
    try:
      events = self.conn.receive_data(data)
    except ProtocolError as e:
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

  def _respond_echo(self, stream_id: int, request_data: RequestData):
    headers = request_data.headers
    body = request_data.data.getvalue().decode('utf-8')

    data = json.dumps({'headers': headers, 'body': body}, indent=4).encode(
        'utf8'
    )

    response_headers = (
        (':status', '200'),
        ('content-type', 'application/json'),
        ('content-length', str(len(data))),
        ('server', 'asyncio-h2'),
    )
    self.conn.send_headers(stream_id, response_headers)
    asyncio.ensure_future(self.send_data(data, stream_id))

  def _respond_404(self, stream_id: int, request_data: RequestData):
    path = request_data.headers[':path']
    response = f'Not Found: {path}\n'
    response_headers = (
        (':status', '404'),
        ('content-type', 'text/plain'),
        ('content-length', str(len(response))),
        ('server', 'asyncio-h2'),
    )
    self.conn.send_headers(stream_id, response_headers)
    asyncio.ensure_future(self.send_data(response.encode(), stream_id))

  def _handle_put(self, stream_id: int, request_data: RequestData):
    if 'etag' not in request_data.headers:
      request_data.headers['etag'] = str(datetime.datetime.now())
    path = request_data.headers[':path']
    _DATA[path] = request_data

    response = f'Received {len(request_data.data.getvalue())} bytes\n'
    response_headers = (
        (':status', '200'),
        ('content-type', 'text/plain'),
        ('content-length', str(len(response))),
        ('server', 'asyncio-h2'),
    )
    self.conn.send_headers(stream_id, response_headers)
    asyncio.ensure_future(self.send_data(response.encode(), stream_id))

  def _handle_get(self, stream_id: int, request_data: RequestData):
    path = request_data.headers[':path']
    if path not in _DATA:
      self._respond_404(stream_id, request_data)
      return

    response = request_data.data.getvalue()
    response_headers = (
        (':status', '200'),
        ('content-type', 'text/plain'),
        ('content-length', str(len(response))),
        ('server', 'asyncio-h2'),
    )
    self.conn.send_headers(stream_id, response_headers)
    asyncio.ensure_future(self.send_data(response, stream_id))

  def _handle_head(self, stream_id: int, request_data: RequestData):
    path = request_data.headers[':path']
    if path not in _DATA:
      self._respond_404(stream_id, request_data)
      return

    response_data = _DATA[path]
    response_headers = (
        (':status', '200'),
        ('content-type', 'text/plain'),
        ('content-length', str(len(response_data.data.getvalue()))),
        ('server', 'asyncio-h2'),
    )
    self.conn.send_headers(stream_id, response_headers)
    asyncio.ensure_future(self.send_data(b'', stream_id))

  def _handle_delete(self, stream_id: int, request_data: RequestData):
    path = request_data.headers[':path']
    if path not in _DATA:
      self._respond_404(stream_id, request_data)
      return

    del _DATA[path]
    self.conn.send_headers(stream_id, [(':status', '200')])
    asyncio.ensure_future(self.send_data(b'', stream_id))

  def stream_complete(self, stream_id: int):
    """When a stream is complete, we can send our response."""
    try:
      request_data = self.stream_data[stream_id]
    except KeyError:
      # Just return, we probably 405'd this already
      return

    method = request_data.headers[':method']
    if method == 'PUT' or method == 'POST':
      self._handle_put(stream_id, request_data)
    elif method == 'DELETE':
      self._handle_delete(stream_id, request_data)
    elif method == 'HEAD':
      self._handle_head(stream_id, request_data)
    elif method == 'GET':
      self._handle_get(stream_id, request_data)
    else:
      self._respond_echo(stream_id, request_data)

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


def run(port, cert_path):
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

  # Serve requests until Ctrl+C is pressed
  print('Serving on {}'.format(server.sockets[0].getsockname()))
  sys.stdout.flush()

  try:
    loop.run_forever()
  except KeyboardInterrupt:
    pass
  finally:
    # Close the server
    server.close()
    loop.run_until_complete(server.wait_closed())
    loop.close()


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--port', type=int, default=0)
  p.add_argument('--cert_path', type=str, default=None)
  v, _ = p.parse_known_args(sys.argv[1:])
  print(f'Starting h2_server with --port={v.port}')
  sys.exit(run(v.port, v.cert_path))
