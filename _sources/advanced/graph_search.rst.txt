.. _graph-search:

Graph-based similarity search
-------------------------------
Graph-based methods use proximity graphs, where nodes represent data vectors and two nodes are connected if they fulfill
a defined property or neighborhood criterion, building on the structure inherent in the data. Search involves starting
at a designated entry point and traversing the graph to get closer and closer to the nearest neighbor with each hop. We
follow the Vamana [SDSK19]_ algorithm for graph building and search.

.. _graph-search-details:

How does the graph search work?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The simplest way to traverse the graph to find 1 approximate nearest neighbor is to do a **greedy search**. At each hop,
the distances from the query to all the neighbors of the current node (i.e., vectors in the current node's adjacency
list) are computed and the closest point is chosen as the next point to be explored. The search ends when the distance to
the query cannot be further reduced by jumping to any of the neighbors of the current node.

**How do we find k neighbors?** To improve the search accuracy and be able to find k nearest neighbors, this greedy search is combined with a **priority
queue**. While traversing the graph, we keep track of the distance from the query to the ``search_window_size``
closest points seen so far (where ``search_window_size`` is the length of the priority queue). At each hop, we choose to
explore next the closest point in the priority queue that has not been visited yet. The search ends when all the
neighbors of the current node are further from the query than the furthest point in the priority queue. This prevents
the search path to diverge too far from the query. A larger ``search_window_size`` implies exploring a larger volume,
improving the accuracy at the cost of a longer search path.

.. _graph-building-details:

How does graph building work?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
First, we :ref:`set the hyper-parameters <graph-build-param-setting>` required to build the graph: ``alpha``,
``graph_max_degree``, ``window_size``, ``max_candidate_pool_size`` (see :py:class:`svs.VamanaBuildParameters` and
:cpp:class:`svs::index::vamana::VamanaBuildParameters` for more details).

.. _graph-building-pseudocode:

Then, the graph is built following the Vamana indexing algorithm [SDSK19]_ as follows:

#. Start from an uninitialized graph **G**.
#. Iterate through all nodes in a random order.

   a. Run the search for node **x** on the current **G**, with the search window size set to ``window_size``, and save the list of visited nodes **C**.
   b. Update **G** by :ref:`pruning <graph-pruning>` **C** to determine the new set of **x**'s neighbors.
   c. Add backward edges (**x**, **x***) for all **x*** in **x**'s out neighbors and prune **x***' edges.

#. Make two passes over the dataset, the first one with the pruning parameter `alpha` =1 and the second one with `alpha` = ``alpha``.
#. Return graph **G** to be used by the search algorithm.

The **pruning rule** limits **x**'s out-neighbors **N** to a maximum of ``graph_max_degree`` as follows:

.. _graph-pruning:

#. Set the list of neighbors candidates **C** = **C** U **N** \\ { **x** }
#. Sort **C** in ascending distance from **x**, and limit **C** to the closest ``max_candidate_pool_size`` neighbors.
#. Initialize **N** to null
#. While **C** is not empty do:

   a. Find **x*** the closest point to **x** in **C**.
   b. Add **x*** to **x**'s out-neighbors list **N**.
   c. If *length* ( **N** ) > ``graph_max_degree`` then break; else remove all points from **C** that are closer to **x*** than **x** by a factor `alpha`.