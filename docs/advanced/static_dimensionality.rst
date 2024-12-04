.. _vamana.h: ../../bindings/python/src/vamana.h
.. _dynamic_vamana.h: ../../bindings/python/src/dynamic_vamana.h
.. _flat.cpp: ../../bindings/python/src/flat.cpp

.. _static-dim:

Static Dimensionality
=====================

SVS supports the option of setting the dimensionality at compile time (static) versus at runtime (dynamic).
Setting it statically is a notable optimization because, as the dimensionality of a dataset is fixed, it presents no
detrimental aspects and it boosts performance by improving the compiler's ability to unroll loops in the similarity
function kernel more extensively. For example, for the standard 96-dimensional dataset `Deep <http://sites.skoltech.ru/compvision/noimi/>`_,
we observe up to a 32% performance speedup when using static versus dynamic dimensionality to search on 100 million points [ABHT23]_.

Uncompressed data
-----------------
**In Python**, to add support to the svs module for static dimensionality for the :ref:`Vamana graph index <vamana_api>`:

1. Define the desired dimensionality specialization in the vamana.h_ file by adding the corresponding line to the ``for_standard_specializations`` template
   indicating the desired query data type, vector data type and dimensionality (see :ref:`supported data types <supported_data_types>`).

   For example, to add static dimensionality support for the 96-dimensional dataset `Deep <http://sites.skoltech.ru/compvision/noimi/>`_,
   for float32-valued queries and base vectors, add the following line:

.. code-block:: cpp

   XN(float,   float, 96);

Or use the following if also want to enable graph building directly from a Numpy array.

.. code-block:: cpp

   X (float,   float, 96, EnableBuild::FromFileAndArray);

2. :ref:`Install svs <install_svs>`.

For the :ref:`Dynamic <dynamic_vamana_api>` and :ref:`Flat <flat_api>` indices follow the same procedure with the
dynamic_vamana.h_ and flat.cpp_ files respectively.

**In C++**

.. collapse:: Click to display

    When building or loading an index, the ``Extent`` template argument of the :cpp:class:`svs::VectorDataLoader` needs to
    be set to the specified dimensionality.

    .. code-block:: cpp

        svs::VectorDataLoader<float, 96>("data_f32.svs")

|
