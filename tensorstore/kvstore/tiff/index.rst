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

Keys are formatted as: ``tile/<ifd>/<row>/<col>``

* ``<ifd>``: The Image File Directory (IFD) index (0-based).
* ``<row>``: Row index for the tile/strip (0-based)
* ``<col>``: Column index for the tile (always 0 for stripped TIFFs)

For example, the key ``tile/0/3/2`` refers to the tile at row 3, column 2 in the first IFD.

Features
--------

* Support for both tiled and stripped TIFF formats
* Multi-page TIFF support via IFD indices
* Handles various bit depths and sample formats

Limitations
-----------

* Writing is not supported (read-only) and not all TIFF features are supported.