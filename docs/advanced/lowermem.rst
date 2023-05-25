.. _lowermem:

Searching with a small memory footprint
---------------------------------------

In large-scale scenarios, the memory requirement for graph-based methods grows quickly. To reduce the memory footprint
we recommend:

#. Use :ref:`vector compression for search <search_with_compression>`. The supported vector compression, :ref:`LVQ <vector_compression>`, reduces the
   memory footprint and improves performance.

#. Build the graph with a small ``graph_max_degree`` (e.g., 32). SVS optimizations enable very high search performance even in graphs
   built with small ``graph_max_degree`` (see :ref:`search_with_reduced_memory_benchs`).

#. Use :ref:`vector compression for graph building <building_with_compressed_vectors>`. The supported vector compression,
   :ref:`LVQ <vector_compression>`, enables graph building with compressed vectors with almost no degradation in search
   accuracy compared to a graph built with full precision vectors [ABHT23]_.
