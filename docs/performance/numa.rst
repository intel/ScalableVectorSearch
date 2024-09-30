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

.. _numa:

NUMA Systems
============

If you are using a server with multiple NUMA domains, and the DRAM available in one socket is enough to run the search
(that is, it should fit at least the graph and the vectors), we recommend using ``numactl`` to restrict the search to
run in a single socket

.. code-block:: sh

    numactl -m 0 -N 0 python example_vamana.py
