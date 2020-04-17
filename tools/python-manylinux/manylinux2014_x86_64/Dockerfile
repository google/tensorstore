ARG BASE=quay.io/pypa/manylinux2014_x86_64

FROM $BASE

LABEL description="A docker image for building portable Python linux binary wheels using modern GCC"
LABEL maintainer="jbms@google.com"

RUN yum -y update && yum -y install \
    devtoolset-9-gcc-c++ \
    && yum clean all

ENV PATH="/opt/rh/devtoolset-9/root/usr/bin:$PATH"
