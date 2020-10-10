Overview
========

A **TensorStore** is an asynchronous view of a multi-dimensional array.  Every
TensorStore is backed by a **driver**, which connects the high-level TensorStore
interface to an underlying data storage mechanism.  Using an appropriate driver,
a TensorStore may be used to access:

- contiguous in-memory arrays
- chunked storage formats like `zarr <https://github.com/zarr-developers/zarr-python>`_, `N5 <https://github.com/saalfeldlab/n5>`_, `Neuroglancer precomputed <https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed>`_, backed by a supported key-value storage system, such as:

  - Google Cloud Storage
  - Local and network filesystems
