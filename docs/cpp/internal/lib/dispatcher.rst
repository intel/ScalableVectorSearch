.. Copyright (C) 2024 Intel Corporation
..
.. This software and the related documents are Intel copyrighted materials,
.. and your use of them is governed by the express license under which they
.. were provided to you ("License"). Unless the License provides otherwise,
.. you may not use, modify, copy, publish, distribute, disclose or transmit
.. this software or the related documents without Intel's prior written
.. permission.
..
.. This software and the related documents are provided as is, with no
.. express or implied warranties, other than those that are expressly stated
.. in the License.

.. _cpp_lib_dispatcher:

Dynamic Dispatcher
==================

A driving philosophy behind SVS is to allow compile-time specialization on as many features
as possible, while still providing generic fallbacks.

This has an interesting interaction with the type-erasure techniques used throughout the
library, namely:

* How to we gather the available specializations for a given type-erased together in one
  place?

* Given a collection of argument values, how do we pick the "best" specialization?

The solution to this is the ``svs::lib::Dispatcher``.

Example with Exposition
-----------------------

Motivation
**********

We begin with a motivating example.
Suppose we have a generic method

.. literalinclude:: ../../../../examples/cpp/dispatcher.cpp
   :language: cpp
   :start-after: [generic-fallback]
   :end-before: [generic-fallback]

That operates on dynamic types ``a_type`` and ``b_type``, a dynamically determined extent
(vector dimension), and takes some type ``Converted`` as the final argument.

Next, suppose we can do better if we hoist ``a_type`` and ``b_type`` into the type domain
for some select types of interest, and perhaps propagate some static extent as well.
So, we define a specialization:

.. literalinclude:: ../../../../examples/cpp/dispatcher.cpp
   :language: cpp
   :start-after: [specialization-1]
   :end-before: [specialization-1]

Finally, maybe we can do *even* better if we can specialize the final argument on a boolean
instead of a string:

.. literalinclude:: ../../../../examples/cpp/dispatcher.cpp
   :language: cpp
   :start-after: [specialization-2]
   :end-before: [specialization-2]

At this point, we have three methods: a generic fallback and two specialized templates.
In our final program, we cannot instantiate our specialized implementations for all possible
types and values as that either would bloat the final binary significantly or simply be
unfeasible.

In an ideal world, we could define a list of selected specializations in a centralized
location, invoke those when our run-time parameters match those specializations, and
invoke the fallback otherwise.

This is where the ``svs::lib::Dispatcher`` comes into play.

Dispatcher
**********

.. literalinclude:: ../../../../examples/cpp/dispatcher.cpp
   :language: cpp
   :start-after: [dispatcher-definition]
   :end-before: [dispatcher-definition]

Above, we see the definition of a ``std::variant`` (corresponding to the final
``std::string`` or ``bool`` arguments of our specializations.
Next, there is the definition of a dispatcher taking four arguments and returning ``void``.

With this dispatcher, we can register our specializations and generic fallback:

.. literalinclude:: ../../../../examples/cpp/dispatcher.cpp
   :language: cpp
   :start-after: [register-methods]
   :end-before: [register-methods]

In the above snippet, we see target registration by passing a reference to the
fully-instantiated specializations and a single instance of the generic method.

.. NOTE::

   Passing C++ functions by reference is an acceptable way to pass the desired dispatch
   target. Mechanically, a method representing the full specialization will be compiled and
   the function reference will decay to a function pointer to this specialization.

   It is also acceptable to pass a lambda directly by value.

   When passing a lambda, it is crucial to ensure that any value captured by reference
   properly outlives the life of the dispatcher.

Upon registration, SVS will check that all source argument types of the dispatcher are
convertible to the argument types of the target by checking for a specialization of
``svs::lib::DispatcherConverter``.
The converter defines match suitability (whether a conversion from source value to
destination type is possible and if so, how "good" that conversion is) and the actual
argument conversion.

SVS already contains rules for converting ``svs::DataType`` to ``svs::lib::Type``, rules
for recursively matching ``std::variant`` types, and conversions between different
applicable reference qualifiers of the same type.

To hook in the custom ``Converted`` type into this system, we can define our own conversion
rules:

.. literalinclude:: ../../../../examples/cpp/dispatcher.cpp
   :language: cpp
   :start-after: [converted-dispatch-conversion-rules]
   :end-before: [converted-dispatch-conversion-rules]

With these rules defined, SVS fill figure out how to convert each alternative in the
variant into a ``Converted`` if needed.

Example Runs
************

Now, a main function can be defined that parses commandline arguments into the dispatch
types and invokes the overload resolution logic in the dispatcher.

.. literalinclude:: ../../../../examples/cpp/dispatcher.cpp
   :language: cpp
   :start-after: [main]
   :end-before: [main]

Possible runs may look like this:

::

    Input: float16 float16 128 false hello
    Output: Generic: float16, float16, dynamic with arg "hello"

    Input: float16 float16 128 false true
    Output: Generic: float16, float16, dynamic with arg "boolean true"

    Input: float16 float16 128 true true
    Output: ANNException (no match found)

    Input: uint32 uint8 128 true hello
    Output: Specialized with string: uint32, uint8, 128 with arg "hello"

    Input: float32 float32 100 false false
    Output: Specialized with flag: float32, float32, dynamic with arg "false"

Automatic Documentation Generation
**********************************

One advantage of grouping all methods together in a single place is that we can use the
documentation feature of the dispatcher to describe all registered methods. The code for
the help message for our example executable is show below

.. literalinclude:: ../../../../examples/cpp/dispatcher.cpp
   :language: cpp
   :start-after: [print-help]
   :end-before: [print-help]

The generated help message might look something like this:

::

    Registered Specializations
    --------------------------
    { type A, type B, Extent, Last Argument }

    { float32, float32, any, all values -- (union alternative 1) }
    { float32, float32, any, all values -- (union alternative 0) }
    { uint32, uint8, 128, all values -- (union alternative 1) }
    { all values, all values, any, all-boolean-values OR all-string-values -- (union alternatives 0, 1) }

API Documentation
-----------------

Classes
*******

.. doxygenclass:: svs::lib::Dispatcher
   :project: SVS
   :members:

.. doxygenclass:: svs::lib::DispatchTarget
   :project: SVS
   :members:

Dispatch API
************

.. doxygenstruct:: svs::lib::DispatchConverter
   :project: SVS

.. doxygenconcept:: svs::lib::DispatchConvertible
   :project: SVS

.. doxygenfunction:: svs::lib::dispatch_match
   :project: SVS

.. doxygenfunction:: svs::lib::dispatch_convert
   :project: SVS

.. doxygenfunction:: svs::lib::dispatch_description
   :project: SVS

.. doxygenvariable:: svs::lib::dispatcher_build_docs
   :project: SVS

.. doxygenvariable:: svs::lib::dispatcher_no_docs
   :project: SVS

Predefined Scores
*****************

.. doxygenvariable:: svs::lib::invalid_match
   :project: SVS

.. doxygenvariable:: svs::lib::perfect_match
   :project: SVS

.. doxygenvariable:: svs::lib::imperfect_match
   :project: SVS

.. doxygenvariable:: svs::lib::implicit_match
   :project: SVS

Helpers
*******

.. doxygenconcept:: svs::lib::ImplicitlyDispatchConvertible
   :project: SVS

.. doxygenstruct:: svs::lib::variant::VariantDispatcher
   :project: SVS
   :members:

Full Example
------------

The full example described at the beginning is given below.

.. literalinclude:: ../../../../examples/cpp/dispatcher.cpp
   :language: cpp
   :start-after: [example-all]
   :end-before: [example-all]
