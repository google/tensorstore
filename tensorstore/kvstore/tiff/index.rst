.. _tiff-kvstore-driver:

``tiff`` Key-Value Store driver
======================================================

The ``tiff`` driver implements support for reading from 
`TIFF <https://en.wikipedia.org/wiki/TIFF>`_ format
files on top of a base key-value store. It provides access to individual tiles or strips
within TIFF images in a standardized key-value format.

.. json:schema:: kvstore/tiff

Example JSON specifications
---------------------------

.. code-block:: json

    { 
      "driver": "tiff",
      "base": "gs://my-bucket/path/to/file.tiff" 
    }

.. code-block:: json
   
    { 
      "driver": "tiff",
      "base": { 
        "driver": "file", 
        "path": "/path/to/image.tiff" 
      } 
    }


Key Format
----------

Keys are formatted as: ``chunk/<ifd>/<index>``

* ``<ifd>``: The Image File Directory (IFD) index (0-based).
* ``<index>``: Linear index for the tile/strip (0-based)

For example, the key ``chunk/0/3`` refers to the third tile/strip in the first IFD.

Features
--------

* Support for both tiled and stripped TIFF formats
* Multi-page TIFF support via IFD indices
* Handles various bit depths and sample formats

Limitations
-----------

* Writing is not supported (read-only) and not all TIFF features are supported.