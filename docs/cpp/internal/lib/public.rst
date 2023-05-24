.. _cpp_lib_public:

Public Library Components
=========================

Exceptions
----------

ALl exceptions thrown by this library that aren't part of the STL's suite of exceptions will be of the type :cpp:class:`svs::ANNException` described below.

.. doxygengroup:: lib_exception
   :project: SVS
   :members:
   :content-only:

Data Types
----------

.. doxygengroup:: lib_public_datatype
   :project: SVS
   :members:
   :content-only:

Static and Dynamic Dimensionality
---------------------------------

Vector abstractions in the library often come with the ability to statically type dimensionality (i.e., the number of elements in a vector).
When static dimensionality is used, the compiler/library can often generate better code resulting in substantial speedups.
Unfortunately, there is no free lunch as specializing functions for specific dimensionalities results in extra compilation time and code generation.
Therefore, we include the option for both static (compile time) and dynamic (run time) dimensionality for internal vectors.

.. doxygengroup:: lib_public_dimensions
   :project: SVS
   :members:
   :content-only:

Types
-----

The library has support for passing either types or lists of types explicitly to some functions.
Often, this works better than relying on positional type parameters of those respective functions as these classes can be passed for any argument and are generally more flexible.

.. doxygengroup:: lib_public_types
   :project: SVS
   :members:
   :content-only:

Example
^^^^^^^

An example of using types is given below.

.. doxygenpage:: example_types_cpp
   :content-only:

Memory Management
-----------------

.. doxygengroup:: lib_public_memory
   :project: SVS
   :members:
   :content-only:

Reading and Writing to Streams
------------------------------

.. doxygenfunction:: svs::lib::write_binary
   :project: SVS

.. doxygengroup:: read_binary_group
   :project: SVS
   :members:
   :content-only:

