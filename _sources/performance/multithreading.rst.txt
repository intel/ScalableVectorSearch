.. _multithreading:

Multithreading
===============

SVS throughput can be highly improved by running multiple queries in parallel, as its performance scales very well with
the number of threads (see :ref:`System utilization - Multithreading <multithreading_scaling>` for benchmarking results). Larger query batch sizes will benefit more
from multi-threaded search.

The number of threads can be specified when loading the index

**In Python**

.. literalinclude:: ../../examples/python/example_vamana.py
   :language: python
   :start-after: [only-loading]
   :end-before: [only-loading]
   :dedent: 4

**In C++**

.. collapse:: Click to display

    .. literalinclude:: ../../examples/cpp/vamana.cpp
       :language: cpp
       :start-after: [Only Loading]
       :end-before: [Only Loading]
       :dedent: 4

|

or at runtime

**In Python**

.. literalinclude:: ../../examples/python/example_vamana.py
   :language: python
   :start-after: [runtime-nthreads]
   :end-before: [runtime-nthreads]
   :dedent: 4

**In C++**

.. collapse:: Click to display

    .. literalinclude:: ../../examples/cpp/vamana.cpp
       :language: cpp
       :start-after: [Set n-threads]
       :end-before: [Set n-threads]
       :dedent: 4

|

Make sure to set the number of threads according to the query batch size. Smaller
batch sizes may benefit from using fewer threads, as the overhead of handling multiple threads may be too large.

.. warning::

    To avoid performance degradation make sure to set the number of threads to 1 if running one query at a time.
