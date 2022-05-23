#!/bin/bash
#
# Adds a url to the tensorstore-bazel-mirror.
gsutil=gsutil
url="${1:?url}"
dest="gs://tensorstore-bazel-mirror/${url#http*//}"
desturl="https://storage.googleapis.com/tensorstore-bazel-mirror/${url#http*//}"
name="$(basename "${dest}")"
wget -O "/tmp/${name}" "${url}" || exit 1
$gsutil cp -n "/tmp/${name}" "${dest}" || exit 1
$gsutil setmeta -h 'Cache-Control:public, max-age=31536000' "${dest}" || exit 1
curl -I "${desturl}"
echo
sha256sum "/tmp/${name}"
echo "${desturl}"
rm "/tmp/${name}" || exit 1
