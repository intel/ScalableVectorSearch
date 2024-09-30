.. Copyright (C) 2023 Intel Corporation
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

.. _compression_for_perf:

Vector Compression
==================

Search performance can be improved by using the Locally-adaptive Vector Quantization (LVQ) [ABHT23]_ approach to
compress the dataset vectors. See :ref:`compression-setting` for information on how to set LVQ parameters.

When the focus is to improve search performance, one should favor the two-level LVQ compression with a small number of bits in the first level
(typically 4 or 8) and a larger number of bits in the second level (typically 8). The best option, however, will depend on
the dataset. **High-dimensional** datasets (>200 dimensions) can largely benefit from LVQ-4x8 or LVQ-8x8 for example.
For **lower dimensional datasets** (<200 dimensions), **one-level** LVQ-8 is often a good choice.
We suggest reviewing the :ref:`SVS + Vector compression (large scale datasets) <benchs-compression-evaluation>` and
:ref:`SVS + Vector compression (small scale datasets) <benchs-compression-evaluation_small_scale>` sections for reference results.

See :ref:`search_with_compression` for details on how to use LVQ for search.
