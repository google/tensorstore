# Copyright 2020 The TensorStore Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Defines a third-party bazel repo for the `apache-beam` Python package."""

load(
    "//third_party:repo.bzl",
    "third_party_python_package",
)
load("//third_party:pypa/avro_python3/workspace.bzl", repo_pypa_avro_python3 = "repo")
load("//third_party:pypa/crcmod/workspace.bzl", repo_pypa_crcmod = "repo")
load("//third_party:pypa/dill/workspace.bzl", repo_pypa_dill = "repo")
load("//third_party:pypa/fastavro/workspace.bzl", repo_pypa_fastavro = "repo")
load("//third_party:pypa/future/workspace.bzl", repo_pypa_future = "repo")
load("//third_party:pypa/grpcio/workspace.bzl", repo_pypa_grpcio = "repo")
load("//third_party:pypa/hdfs/workspace.bzl", repo_pypa_hdfs = "repo")
load("//third_party:pypa/httplib2/workspace.bzl", repo_pypa_httplib2 = "repo")
load("//third_party:pypa/mock/workspace.bzl", repo_pypa_mock = "repo")
load("//third_party:pypa/numpy/workspace.bzl", repo_pypa_numpy = "repo")
load("//third_party:pypa/oauth2client/workspace.bzl", repo_pypa_oauth2client = "repo")
load("//third_party:pypa/protobuf/workspace.bzl", repo_pypa_protobuf = "repo")
load("//third_party:pypa/pyarrow/workspace.bzl", repo_pypa_pyarrow = "repo")
load("//third_party:pypa/pydot/workspace.bzl", repo_pypa_pydot = "repo")
load("//third_party:pypa/pymongo/workspace.bzl", repo_pypa_pymongo = "repo")
load("//third_party:pypa/python_dateutil/workspace.bzl", repo_pypa_python_dateutil = "repo")
load("//third_party:pypa/pytz/workspace.bzl", repo_pypa_pytz = "repo")
load("//third_party:pypa/typing_extensions/workspace.bzl", repo_pypa_typing_extensions = "repo")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    repo_pypa_avro_python3()
    repo_pypa_crcmod()
    repo_pypa_dill()
    repo_pypa_fastavro()
    repo_pypa_future()
    repo_pypa_grpcio()
    repo_pypa_hdfs()
    repo_pypa_httplib2()
    repo_pypa_mock()
    repo_pypa_numpy()
    repo_pypa_oauth2client()
    repo_pypa_protobuf()
    repo_pypa_pyarrow()
    repo_pypa_pydot()
    repo_pypa_pymongo()
    repo_pypa_python_dateutil()
    repo_pypa_pytz()
    repo_pypa_typing_extensions()
    maybe(
        third_party_python_package,
        name = "pypa_apache_beam",
        target = "apache_beam",
        requirement = "apache-beam==2.20.0",
        deps = [
            "@pypa_avro_python3//:avro_python3",
            "@pypa_crcmod//:crcmod",
            "@pypa_dill//:dill",
            "@pypa_fastavro//:fastavro",
            "@pypa_future//:future",
            "@pypa_grpcio//:grpcio",
            "@pypa_hdfs//:hdfs",
            "@pypa_httplib2//:httplib2",
            "@pypa_mock//:mock",
            "@pypa_numpy//:numpy",
            "@pypa_oauth2client//:oauth2client",
            "@pypa_protobuf//:protobuf",
            "@pypa_pyarrow//:pyarrow",
            "@pypa_pydot//:pydot",
            "@pypa_pymongo//:pymongo",
            "@pypa_python_dateutil//:python_dateutil",
            "@pypa_pytz//:pytz",
            "@pypa_typing_extensions//:typing_extensions",
        ],
    )
