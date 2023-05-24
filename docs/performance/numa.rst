.. _numa:

NUMA Systems
============

If you are using a server with multiple NUMA domains, and the DRAM available in one socket is enough to run the search
(that is, it should fit at least the graph and the vectors), we recommend using ``numactl`` to restrict the search to
run in a single socket

.. code-block:: sh

    numactl -m 0 -N 0 python example_vamana.py
