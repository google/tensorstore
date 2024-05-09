#include <iostream>
#include <streambuf>
#include <string>
#include <string_view>


#include <aws/core/Aws.h>
#include <aws/core/auth/AWSAuthSigner.h>
#include <aws/core/http/standard/StandardHttpRequest.h>
#include <aws/core/http/URI.h>

#include "absl/strings/cord.h"

#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/kvstore/s3_sdk/s3_context.h"

namespace tensorstore {
namespace internal_kvstore_s3 {

// Make an absl::Cord look like a streambuf
class CordStreambuf : public std::streambuf {
public:
  CordStreambuf(const absl::Cord& cord) : cord_(cord), current_(cord_.char_begin()) {
    setg(nullptr, nullptr, nullptr);
  }

protected:
  // Refill the get area of the buffer
  int_type underflow() override {
    if (current_ == cord_.char_end()) {
      return traits_type::eof();
    }

    // Set buffer pointers for the next character
    setg(const_cast<char*>(&*current_),
         const_cast<char*>(&*current_),
         const_cast<char*>(&*std::next(current_)));

    return traits_type::to_int_type(*current_++);
  }

private:
    const absl::Cord& cord_;
    absl::Cord::CharIterator current_;
};

// Make an absl::Cord look like an iostream
class CordIOStream : public std::iostream {
public:
  CordIOStream(const absl::Cord& cord) : std::iostream(&buffer_), buffer_(cord) {
    rdbuf(&buffer_);
  }

private:
  CordStreambuf buffer_;
};

class AwsHttpRequestAdapter : public Aws::Http::Standard::StandardHttpRequest {
private:
  static Aws::Http::HttpMethod FromStringMethod(std::string_view method) {
    if(method == "GET") {
      return Aws::Http::HttpMethod::HTTP_GET;
    } else if (method == "PUT") {
      return Aws::Http::HttpMethod::HTTP_PUT;
    } else if (method == "HEAD") {
      return Aws::Http::HttpMethod::HTTP_HEAD;
    } else if (method == "DELETE") {
      return Aws::Http::HttpMethod::HTTP_DELETE;
    } else if (method == "POST") {
      return Aws::Http::HttpMethod::HTTP_POST;
    } else if (method == "PATCH") {
      return Aws::Http::HttpMethod::HTTP_PATCH;
    } else {
      // NOTE: return an error
      return Aws::Http::HttpMethod::HTTP_GET;
    }
  }

public:
  AwsHttpRequestAdapter(std::string_view method, std::string endpoint_url) :
    Aws::Http::Standard::StandardHttpRequest(Aws::Http::URI(Aws::String(endpoint_url)),
    FromStringMethod(method)) {}
};

class NewS3RequestBuilder {
public:
  NewS3RequestBuilder(std::string_view method, std::string endpoint_url) :
    request_(method, endpoint_url) {}

  NewS3RequestBuilder & AddBody(const absl::Cord & body) {
    // NOTE: eliminate allocation
    auto cord_adapter = std::make_shared<CordIOStream>(body);
    request_.AddContentBody(cord_adapter);
    return *this;
  }

  NewS3RequestBuilder & AddHeader(std::string_view header) {
    auto delim_pos = header.find(':');
    assert(delim_pos != std::string_view::npos);
    // NOTE: string copies
    request_.SetHeaderValue(std::string(header.substr(0, delim_pos)).c_str(),
                            Aws::String(header.substr(delim_pos + 1)));
    return *this;
  }

  NewS3RequestBuilder & AddQueryParameter(std::string key, std::string value) {
    // Note: string copies
    request_.AddQueryStringParameter(key.c_str(), Aws::String(value));
    return *this;
  }

  internal_http::HttpRequest BuildRequest(AwsContext ctx) {
    auto signer = Aws::Client::AWSAuthV4Signer(ctx.cred_provider_, "s3", "us-east-1");
    assert(!request_.HasAuthorization());
    auto succeeded = signer.SignRequest(request_, true);
    assert(succeeded);
    assert(request_.HasAuthorization());
    auto method = Aws::Http::HttpMethodMapper::GetNameForHttpMethod(request_.GetMethod());
    auto aws_headers = request_.GetHeaders();

    std::vector<std::string> headers;
    headers.reserve(aws_headers.size());

    for(auto & pair: aws_headers) {
      headers.emplace_back(absl::StrFormat("%s: %s", pair.first, pair.second));
    }

    return internal_http::HttpRequest{
      std::move(method),
      std::string(request_.GetURIString(true)),
      "",
      headers};
  }

public:
  std::shared_ptr<Aws::IOStream> body_;
  AwsHttpRequestAdapter request_;
};

}  // namespace internal_kvstore_s3
}  // namespace tensorstore
