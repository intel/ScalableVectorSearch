.. _cpp_orchestrators:

Indexes
=======

The main index interfaces are listed here and in the next sections.
Refer to this table when making decisions about which index to use.

* :cpp:class:`svs::Vamana` - A graph-based similarity search engine suitable for high recall, high throughput, and low latency.
* :cpp:class:`svs::Flat` - An exhaustive search engine.
  Provides precise results at the expense of low throughput and high latency for large datasets.

Contents
--------

.. toctree::
   vamana.rst
   flat.rst

Compatible Loaders
------------------

The table below lists the compatibility between different index and the different kinds of dataset (e.g., uncompressed, lvq etc).

.. csv-table:: Index and Loader Compatibility Matrix
   :file: loader-compatibility.csv
   :header-rows: 1

.. NOTE::

    As an implementation detail, the indexes here are type-erased instances of more specialized, heavily templated implementations.
    These low level implementations generally have a similar API to the top level indexes and can be more expressive as functions calls do not have to cross a virtual function boundary.
    However, this specialization results in a different concrete type for each implementation which can be undesireable.
    More documentation regarding the low level implementation will be available in the future.
