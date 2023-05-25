.. _distance_concept:

Distance Computation
====================

Functor are used to implement and tailor distance computations in a number of ways.
The library uses functors (i.e., structs) for a number of reasons summarised below.
In general, the function :cpp:func:`svs::distance::compute` is used to compute the similarity between queries on the left and dataset elements on the right.

* Often, distance computation needs to perform some pre-processing per query inorder to
  execute efficiently. Using structs allows for this behavior.
* If structs are empty, than the extra overhead introduced (e.g., space overhead) by them
  is negligible compared to function references or pointers.
* Overloading operations for structs is easier than function references.
* Using functors allows for fancier options like type-erasure to reduce generated code if
  desired.

In particular, bullet point (1) deserves a little explanation.
An efficient implementation for similarity metric such as cosine-similarity may involve either computing the norm of the query (externally supplied vector) or normalizing the vector.
Both cases require some local scratch space for either storing the computed norm or the normalized query (to maintain const-correctness of the externally supplied query).
Using a distance functor allows this scratch space to exist.
A single struct can then be re-used to process multiple queries in series or copied to process queries in parallel.

On the other hand, not all distance implementations (such as L-p distances or inner-product) may not require such state.
This leads to the idea of argument fixing.

This is a test :cpp:func:`compute` of cross referencing.

Argument Fixing
^^^^^^^^^^^^^^^

To support functors both requiring and not needing query preprocessing, the function :cpp:func:`svs::distance::maybe_fix_argument` should be called on a query ``a`` before any distance computations requiring ``a`` as the left-hand argument.
Note that the right-hand argument ``b`` can vary at will.

A distance function ``MyFunctor`` may opt-in to argument fixing by defining the method ``fix_argument`` as follows:

.. code-block:: cpp

   struct MyStruct {
       // Boiler-plate
       template<typename QueryType>
       void fix_argument(const QueryType& query) { /*impl*/ }
   };

The possible values of ``QueryType`` should be restricted as appropriate.

API Documentation
^^^^^^^^^^^^^^^^^

.. doxygengroup:: distance_public
   :project: SVS
   :members:
   :content-only:
