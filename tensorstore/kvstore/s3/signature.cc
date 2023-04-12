#include <algorithm>
#include <set>

#include "absl/log/absl_check.h"
#include "absl/strings/ascii.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "absl/time/time.h"

#include "tensorstore/kvstore/s3/signature.h"
#include "tensorstore/internal/ascii_utils.h"
#include "tensorstore/internal/digest/sha256.h"
#include "tensorstore/util/result.h"

#include <openssl/evp.h>
#include <openssl/hmac.h>

using ::tensorstore::internal::IntToHexDigit;
using ::tensorstore::internal::PercentEncodeReserved;
using ::tensorstore::internal::AsciiSet;
using ::tensorstore::internal::SHA256Digester;
using ::tensorstore::internal::ParseGenericUri;
using ::tensorstore::internal::ParsedGenericUri;

namespace tensorstore {
namespace internal_storage_s3 {

/// Size of HMAC (size of SHA256 digest).
constexpr static size_t kHmacSize = 32;

void ComputeHmac(std::string_view key, std::string_view message, unsigned char (&hmac)[kHmacSize]){
    unsigned int md_len = kHmacSize;
    // Computing HMAC should never fail.
    ABSL_CHECK(HMAC(EVP_sha256(),
                    reinterpret_cast<const unsigned char*>(key.data()),
                    key.size(),
                    reinterpret_cast<const unsigned char*>(message.data()),
                    message.size(), hmac, &md_len) &&
               md_len == kHmacSize);
}

void ComputeHmac(unsigned char (&key)[kHmacSize], std::string_view message, unsigned char (&hmac)[kHmacSize]){
    unsigned int md_len = kHmacSize;
    // Computing HMAC should never fail.
    ABSL_CHECK(HMAC(EVP_sha256(), key, kHmacSize,
                    reinterpret_cast<const unsigned char*>(message.data()),
                    message.size(), hmac, &md_len) &&
               md_len == kHmacSize);
}


namespace {

// See description of function UriEncode at this URL
// https://docs.aws.amazon.com/AmazonS3/latest/API/sig-v4-header-based-auth.html
constexpr AsciiSet kUriUnreservedChars{
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "-._~"};

constexpr AsciiSet kUriObjectKeyReservedChars{
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "/-._~"};
}

std::string UriEncode(std::string_view src) {
    std::string dest;
    PercentEncodeReserved(src, dest, kUriUnreservedChars);
    return dest;
}

std::string UriObjectKeyEncode(std::string_view src) {
    std::string dest;
    PercentEncodeReserved(src, dest, kUriObjectKeyReservedChars);
    return dest;
}

/// https://docs.aws.amazon.com/AmazonS3/latest/API/sig-v4-header-based-auth.html
Result<std::string> CanonicalRequest(
    std::string_view http_method,
    const ParsedGenericUri & uri,
    const std::map<std::string, std::string> & headers,
    std::string_view payload_hash)
{

    // Convert headers and query strings to lower case.
    // std::map sorts them implicitly
    std::map<std::string, std::string> lower_headers;
    std::map<std::string, std::string> lower_query;

    for(auto & key_value: headers) {
        lower_headers.insert({absl::AsciiStrToLower(key_value.first), key_value.second});
    }

    if(!uri.query.empty()) {
        std::vector<std::string> query_bits = absl::StrSplit(uri.query, "&");

        for(auto & query_bit: query_bits) {
            std::vector<std::string> key_value = absl::StrSplit(query_bit, "=");

            if(key_value.size() == 1) {
                lower_query.insert({absl::AsciiStrToLower(key_value[0]), ""});
            } else if(key_value.size() == 2) {
                lower_query.insert({absl::AsciiStrToLower(key_value[0]), key_value[1]});
            } else {
                return absl::InvalidArgumentError(
                    absl::StrCat(query_bit, " in query string ", uri.query, " does not conform to key=value"));
            }
        }
    }

    size_t end_of_bucket = uri.authority_and_path.find('/');

    if(end_of_bucket == std::string_view::npos) {
        return absl::InvalidArgumentError(
            absl::StrCat(uri.authority_and_path, " does not contain a path"));
    }

    absl::Cord cord;
    cord.Append(http_method);
    cord.Append("\n");
    cord.Append(UriObjectKeyEncode(uri.authority_and_path.substr(end_of_bucket)));
    cord.Append("\n");

    // Query string
    for(auto [it, first] = std::tuple{lower_query.begin(), true}; it != lower_query.end(); ++it, first=false) {
        if(!first) {
            cord.Append("&");
        }
        cord.Append(UriEncode(it->first));
        cord.Append("=");
        cord.Append(UriEncode(it->second));
    }

    cord.Append("\n");

    // Headers
    for(auto it = lower_headers.begin(); it != lower_headers.end(); ++it) {
        cord.Append(it->first);
        cord.Append(":");
        cord.Append(absl::StripAsciiWhitespace(it->second));
        cord.Append("\n");
    }

    cord.Append("\n");

    // Signed headers
    for(auto [it, first] = std::tuple{lower_headers.begin(), true}; it != lower_headers.end(); ++it, first=false) {
        if(!first) {
            cord.Append(";");
        }

        cord.Append(it->first);
    }

    cord.Append("\n");
    cord.Append(payload_hash);

    std::string result;
    absl::CopyCordToString(cord, &result);
    return result;
}

/// https://docs.aws.amazon.com/AmazonS3/latest/API/sig-v4-header-based-auth.html
std::string SigningString(
    std::string_view canonical_request,
    const absl::Time & time,
    std::string_view aws_region)
{
    absl::Cord cord;
    absl::TimeZone utc = absl::UTCTimeZone();

    cord.Append("AWS4-HMAC-SHA256\n");
    cord.Append(absl::FormatTime("%Y%m%dT%H%M%SZ", time, utc));
    cord.Append("\n");
    cord.Append(absl::FormatTime("%Y%m%d", time, utc));
    cord.Append("/");
    cord.Append(aws_region);
    cord.Append("/");
    cord.Append("s3");
    cord.Append("/");
    cord.Append("aws4_request");
    cord.Append("\n");

    SHA256Digester sha256;
    sha256.Write(canonical_request);
    cord.Append(sha256.HexDigest(false));

    std::string result;
    absl::CopyCordToString(cord, &result);
    return result;
}

std::string Signature(
    std::string_view aws_secret_access_key,
    std::string_view aws_region,
    const absl::Time & time,
    std::string_view signing_string)
{
    absl::TimeZone utc = absl::UTCTimeZone();

    unsigned char date_key[kHmacSize];
    unsigned char date_region_key[kHmacSize];
    unsigned char date_region_service_key[kHmacSize];
    unsigned char signing_key[kHmacSize];
    unsigned char final_key[kHmacSize];

    ComputeHmac(absl::StrFormat("AWS4%s",aws_secret_access_key), absl::FormatTime("%Y%m%d", time, utc), date_key);
    ComputeHmac(date_key, aws_region, date_region_key);
    ComputeHmac(date_region_key, "s3", date_region_service_key);
    ComputeHmac(date_region_service_key, "aws4_request", signing_key);
    ComputeHmac(signing_key, signing_string, final_key);

    std::string result(2 * kHmacSize, '0');

    for(int i=0; i < kHmacSize; ++i) {
        result[2*i + 0] = IntToHexDigit(final_key[i] / 16, false);
        result[2*i + 1] = IntToHexDigit(final_key[i] % 16, false);
    }

    return result;
}

std::string Authorizationheader(
    const std::string & aws_access_key,
    const std::string & aws_region,
    const std::map<std::string, std::string> & headers,
    const absl::Time & time,
    const std::string & signature)
{
    absl::TimeZone utc = absl::UTCTimeZone();
    absl::Cord cord;

    cord.Append(absl::FormatTime("%Y%m%dT%H%M%SZ", time, utc));
    cord.Append("\n");
    cord.Append(absl::FormatTime("%Y%m%d", time, utc));

    cord.Append("AWS4-HMAC-SHA256 Credential=");
    cord.Append(aws_access_key);
    cord.Append("/");
    cord.Append(absl::FormatTime("%Y%m%d", time, utc));
    cord.Append("/");
    cord.Append(aws_region);
    cord.Append("s3/aws4_request,");

    std::map<std::string, std::string> lower_headers;

    for(auto & key_value: headers) {
        lower_headers.insert({absl::AsciiStrToLower(key_value.first), key_value.second});
    }

    cord.Append("SignedHeaders=");

    for(auto it = lower_headers.begin(); it != lower_headers.end();) {
        cord.Append(it->first);
        if(it == lower_headers.end()) {
            break;
        } else {
            cord.Append(";");
        }
    }

    cord.Append(",");
    cord.Append("Signature=");
    cord.Append(signature);

    std::string result;
    absl::CopyCordToString(cord, &result);
    return result;
}

} // namespace tensorstore
} // namespace internal_storage_s3
