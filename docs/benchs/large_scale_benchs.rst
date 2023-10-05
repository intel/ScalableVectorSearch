.. _large_scale_benchs:

Large Scale Benchmarks
**********************
We present here the results of an exhaustive evaluation, comparing SVS to other implementations as well as evaluating
the performance of different SVS flavors (with different data types, vector compression) for large scale datasets.

.. contents::
   :local:
   :depth: 2

.. _system_setup_large_scale_benchs:

System Setup and Datasets
=========================

We run our experiments on a 3rd generation Intel\ |reg| Xeon\ |reg| Platinum 8380 CPU @2.30GHz with
40 cores (single socket), equipped with 1TB DDR4 memory per socket @3200MT/s speed,  running Ubuntu 22.04. [#ft1]_ [#ft3]_

We use ``numactl`` to ran all experiments in a single socket (see :ref:`numa` for details).

We use the ``hugeadm`` Linux utility to :ref:`preallocate a sufficient number of 1GB huge pages <hugepages>` for each algorithm.
SVS explicitly uses huge pages to reduce the virtual memory overheads.
For a fair comparison, we run other methods with system flags enabled to automatically use huge pages for large allocations.

We consider datasets that are large scale because of their total footprint (see :ref:`datasets` for details):

* deep-96-1B (96 dimensions, 1 billion points)
* sift-128-1B (128 dimensions, 1 billion points)
* t2i-200-100M (200 dimensions, 100 million points)
* DPR-768-10M (768 dimensions, 10 million points)

Comparison to Other Implementations
===================================

.. contents::
   :local:
   :depth: 1


.. _search_with_reduced_memory_benchs:

Search with Reduced Memory Footprint
------------------------------------

In large-scale scenarios, the memory requirement grows quickly, in particular for graph-based methods. This makes these
methods expensive, as the system cost is dominated by the total DRAM price. We compare here the performance, for different
memory footprint regimes, of SVS vs. four widely adopted approaches: Vamana [SDSK19]_, HSNWlib [MaYa18]_, FAISS-IVFPQfs
[JoDJ19]_, and ScaNN [GSLG20]_. [#ft2]_ See :ref:`param_setting_bench_large_scale_low_memory_regime` for details on
the evaluated configurations and the version of the code used for each method.

|

.. image:: ../figs/SVS_performance_memoryfootprint.png
   :width: 600
   :align: center
   :alt: SVS performance vs memory footprint.

|

SVS exploits the power of graph-based similarity search and vector compression [ABHT23]_ to enable high-throughput and
high-accuracy with a small memory footprint. The figure shows the results for a standard billion scale dataset
(`Deep-1B <http://sites.skoltech.ru/compvision/noimi/>`_, 1 billion 96-dimensional vectors) for a search accuracy of 0.9
10-recall at 10. For graph-based methods, the memory footprint is a function of the ``graph_max_degree`` (R in the figure)
and the vector footprint.

With a memory footprint of only 245 GiB, SVS delivers 126k queries-per-second, that is 2.3x, 2.2x, 20.7x, and 43.6x more
throughput with 3.0x, 3.3x, 1.7x, and 1.8x lower memory than the current leaders,
Vamana, HSNWlib, FAISS-IVFPQfs, and ScaNN respectively. With the highest-throughput
configuration, SVS achieves 336k queries-per-second outperforming the second-highest by 5.8x with a 1.4x lower memory footprint (596GiB).

.. _param_setting_bench_large_scale_low_memory_regime:

Parameters Setting
^^^^^^^^^^^^^^^^^^^

We used the following versions of each method:
SVS `commit ad821d8 <https://github.com/IntelLabs/ScalableVectorSearch/commit/ad821d8c94cb69a67c8744b98ee1c79d3e3a299c>`_,
Vamana `commit 647f68f <https://github.com/microsoft/DiskANN/commit/647f68fe5aa7b45124ae298c219fe909d46a1834>`_,
HNSWlib `commmit 4b2cb72 <https://github.com/nmslib/hnswlib/commit/4b2cb72c3c1bbddee55535ec6f360a0b2e40a81e>`_,
ScaNN `commit d170ac5 <https://github.com/google-research/google-research/commit/d170ac58ce1d071614b2813b056afa292f5e490c>`_,
and FAISS-IVFPQfs `commit 19f7696 <https://github.com/facebookresearch/faiss/commit/19f7696deedc93615c3ee0ff4de22284b53e0243>`_.

For **SVS** and **Vamana**, we build Vamana graphs with: ``graph_max_degree`` = 32, 64, 126 and ``alpha`` = 1.2. SVS
is used with LVQ-8 compressed vectors, with vectors padded to half cache lines (``padding`` = 32,
see :ref:`compression-setting` for details).
For **HSNWlib**, we build graphs with a window search size of 200 and ``graph_max_degree`` = 32, 64, 96 (this corresponds
to M=16, 32, 48 in HSNW notation). We had to reduce ``graph_max_degree`` from 128 to 96 to fit the working set size in
1TB memory.

For **FAISS-IVFPQfs**, we use ``nlist`` = 32768 and ``nbins`` :math:`=48`.
Re-ranking is enabled, and at runtime we sweep ``nprobe`` :math:`=[1,5,10,50,100,20]` and  ``k for re-ranking`` :math:`= [0,10,100,1000]`.
For **ScaNN**, we use the recommended parameters setting: ``n_leaves`` = :math:`\sqrt{n}`, ``avq_threshold`` = 0.2,
``dims_per_block`` = 2 (where :math:`n` is the number of vectors in the dataset), as that is the best among several
evaluated settings and vary the runtime parameters (``leaves_to_search`` = [2-1000], ``reorder`` = [20-1000]).
For FAISS-IVFPQfs and ScaNN, which follow the same index design, the memory footprint is almost constant for different
considered parameter settings.



High-throughput Regime
----------------------

In the high-throughput regime, all methods are set assuming high throughput is the main priority and memory availability
is not a major issue. We compare SVS to four widely adopted approaches: Vamana [SDSK19]_, HSNWlib [MaYa18]_, FAISS-IVFPQfs
[JoDJ19]_, and ScaNN [GSLG20]_. [#ft2]_ See :ref:`param_setting_bench_large_scale_high_throughput_regime`
for details on the evaluated configurations and the version of the code used for each method.

Results summary:

* **SVS shows a large performance advantage across recall values for billion scale datasets** with Euclidean distance
  (see results for deep-96-1B and sift-128-1B below).

* For high-dimensional datasets that require inner product, SVS has a significant performance advantage across recall values
  for query batch size 128, and up to recall 0.95 for batch size 10k (see results for t2i-200-100M and DPR-768-10M below).

* For a search accuracy of 0.9 10-recall at 10, SVS achieves

    * **6.5x and 5.4x higher throughput** over the closest competitor for **deep-96-1B** with query batch sizes 10k and 128 respectively.
    * **3.4x and 4.0x higher throughput** over the closest competitor for **sift-128-1B** (uint8-valued vectors) with query batch sizes 10k and 128 respectively.
    * **1.8x and 3.2x higher throughput** over the closest competitor for **DPR-768-10M** with query batch sizes 10k and 128 respectively.
    * **2.0x higher throughput** over the closest competitor for **t2i-200-100M**.

**Click on the triangles** to see the throughput vs recall curves for each dataset.

.. collapse:: deep-96-1B

    Results for the deep-96-1B dataset

    .. image:: ../figs/bench_largeScale_bothBatchSz_deep-1B.png
       :width: 800
       :alt: deep-96-1B benchmarking results

.. collapse:: sift-128-1B

    Results for the sift-128-1B dataset

    .. image:: ../figs/bench_largeScale_bothBatchSz_bigann-1B.png
       :width: 800
       :alt: sift-128-1B benchmarking results

.. collapse:: t2i-200-100M

    Results for the t2i-200-100M dataset

    .. image:: ../figs/bench_largeScale_bothBatchSz_text2image-100M.png
       :width: 800
       :alt: t2i-200-100M benchmarking results

.. collapse:: DPR-768-10M

    Results for the DPR-768-10M dataset

    .. image:: ../figs/bench_largeScale_bothBatchSz_dpr-10M.png
       :width: 800
       :alt: DPR-768-10M benchmarking results

|

.. _param_setting_bench_large_scale_high_throughput_regime:

Parameters Setting
^^^^^^^^^^^^^^^^^^^

We used the following versions of each method:
Vamana `commit 647f68f <https://github.com/microsoft/DiskANN/commit/647f68fe5aa7b45124ae298c219fe909d46a1834>`_,
HNSWlib `commmit 4b2cb72 <https://github.com/nmslib/hnswlib/commit/4b2cb72c3c1bbddee55535ec6f360a0b2e40a81e>`_,
ScaNN `commit d170ac5 <https://github.com/google-research/google-research/commit/d170ac58ce1d071614b2813b056afa292f5e490c>`_,
and FAISS-IVFPQfs `commit 19f7696 <https://github.com/facebookresearch/faiss/commit/19f7696deedc93615c3ee0ff4de22284b53e0243>`_.

For **SVS** and **Vamana**, we use the following parameter setting to build Vamana graphs for all the datasets:

* ``graph_max_degree`` = 128 (we use ``graph_max_degree`` = 126 for deep-96-1B),
* ``alpha`` = 1.2 and ``alpha`` =  0.95 for Euclidean distance and inner product respectively.

For SVS, we include various LVQ settings (LVQ-8, LVQ-4x4, LVQ-4x8, and LVQ8x8) as well as float16 and float32 encodings.
LVQ-compressed vectors are padded to half cache lines (``padding`` = 32).

For **HSNWlib**, we build all graphs with a window search size of 200 and ``graph_max_degree`` = 128 (this corresponds
to M=64 in HSNW notation), except deep-96-1B for which we must reduce ``graph_max_degree`` to 96 to fit the
working set size in 1TB memory.

For **FAISS-IVFPQfs**, we pre-build indices with ``nlist`` = 32768 and ``nbins`` :math:`=d/2` (where :math:`d` is the dataset dimensionality)
for the 1 billion scale datasets deep-96-1B and sift-128-1B. For t2i-200-100M and DPR-768-10M, indices are built on the fly
with combinations of ``nlist`` :math:`=[512, 1024, 4096, 8192]` and ``nbins`` :math:`=[d/4, d/2, d]`.
Re-ranking is enabled, and at runtime we sweep ``nprobe`` :math:`=[1,5,10,50,100,20]` and  ``k for re-ranking`` :math:`= [0,10,100,1000]`.

For **ScaNN**, we use the recommended parameters setting: ``n_leaves`` = :math:`\sqrt{n}`, ``avq_threshold`` = 0.2,
``dims_per_block`` = 2 (where :math:`n` is the number of vectors in the dataset) for the billion scale datasets
(deep-96-1B and sift-128-1B), as that is the best among several evaluated settings. For t2i-200-100M and DPR-768-10M we evaluate
different parameter settings (see Table below). For all dataests we vary the runtime parameters
(``leaves_to_search`` = [2-1000], ``reorder`` = [20-1000]).

+---------------------------------------------------------------------------------------------------------------+
|                                          **ScaNN parameter setting**                                          |
+-------------------------------------------------------+-------------------------------------------------------+
|                    **t2i-200-100M**                   |                    **DPR-768-10M**                    |
+--------------+-------------------+--------------------+--------------+-------------------+--------------------+
| **n_leaves** | **avq_threshold** | **dims_per_block** | **n_leaves** | **avq_threshold** | **dims_per_block** |
+--------------+-------------------+--------------------+--------------+-------------------+--------------------+
|     2000     |        0.2        |          1         |     1000     |        0.55       |          1         |
+--------------+-------------------+--------------------+--------------+-------------------+--------------------+
|     5000     |        0.15       |          3         |     2000     |        0.2        |          1         |
+--------------+-------------------+--------------------+--------------+-------------------+--------------------+
|     10000    |        0.2        |          2         |     3162     |        0.2        |          2         |
+--------------+-------------------+--------------------+--------------+-------------------+--------------------+
|     20000    |        0.2        |          2         |     6000     |        0.2        |          2         |
+--------------+-------------------+--------------------+--------------+-------------------+--------------------+

In all cases where several parameter settings are evaluated, the results show the corresponding Pareto lines.

.. _benchs-compression-evaluation:

SVS + Vector Compression
========================

We show here how the :ref:`LVQ <vector_compression>` vector compression can boost SVS performance relative to using float32 or float16 encoded vectors.
The :ref:`best LVQ flavor <compression-setting>` (whether one or two levels, and the number of bits used to encode each level) depends on the dataset and
the memory footprint restrictions. The results below can serve as reference for datasets of similar dimensionality / cardinality.

The memory-footprint ratio (MR) measures the space occupied by the graph (with ``graph_max_degree`` = 128) and the
float32-valued vectors relative to the space occupied by the graph and the LVQ-compressed vectors. As shown in the table below,
for larger dimensionalities (d = 768, DPR-768-10M dataset), LVQ highly reduces the memory requirements achieving a large MR,
and the additional bandwidth reduction from LVQ-4x4 and LVQ-4x8 provides a significant performance boost over LVQ-8.

+--------------------+---------------------+---------------------+---------------------+
|                    | **deep-96-1B**      | **t2i-200-100M**    | **DPR-768-10M**     |
+--------------------+----------+----------+----------+----------+----------+----------+
| **w.r.t. float32** | **QPS**  | **MR**   | **QPS**  | **MR**   | **QPS**  | **MR**   |
+--------------------+----------+----------+----------+----------+----------+----------+
| **float16**        | 2.1x     | 1.3x     | 1.9x     | 1.4x     | 1.7x     | 1.8x     |
+--------------------+----------+----------+----------+----------+----------+----------+
| **LVQ-8**          | **2.6x** | **1.4x** | 2.9x     | **1.8x** | 3.1x     | **2.7x** |
+--------------------+----------+----------+----------+----------+----------+----------+
| **LVQ-4x4**        | 2.3x     | **1.4x** | 2.2x     | **1.8x** | 4.3x     | **2.7x** |
+--------------------+----------+----------+----------+----------+----------+----------+
| **LVQ-4x8**        | 2.5x     | 1.3x     | **3.1x** | 1.6x     | **4.7x** | 2.1x     |
+--------------------+----------+----------+----------+----------+----------+----------+

Ablation Results
----------------

**Click on the triangles** to see the throughput vs recall curves comparing SVS with several LVQ settings, as well as float32
and float16 encodings, for each dataset.

.. collapse:: deep-96-1B

    Results for the deep-96-1B dataset

    .. image:: ../figs/bench_largeScale_SVS_ablation_deep-1B.png
       :width: 800
       :alt: deep-96-1B compression evaluation results

.. collapse:: deep-96-100M

    Results for the deep-96-100M dataset

    .. image:: ../figs/bench_largeScale_SVS_ablation_deep-100M.png
       :width: 800
       :alt: deep-96-100M compression evaluation results

.. collapse:: t2i-200-100M

    Results for the t2i-200-100M dataset

    .. image:: ../figs/bench_largeScale_SVS_ablation_text2image-100M.png
       :width: 800
       :alt: t2i-200-100M compression evaluation results

.. collapse:: DPR-768-10M

    Results for the DPR-768-10M dataset

    .. image:: ../figs/bench_largeScale_SVS_ablation_dpr-10M.png
       :width: 800
       :alt: DPR-768-10M compression evaluation results

|

Ablation + comparison to other methods
--------------------------------------

**Click on the triangles** to see the throughput vs recall curves comparing the highest performing SVS-LVQ setting
(a Pareto curve built with results from LVQ-8, LVQ-4x4, LVQ4x8 and LVQ8x8), SVS using float32 and float16 encodings,
as well as other methods for each dataset.

.. collapse:: deep-96-1B

    Results for the deep-96-1B dataset

    .. image:: ../figs/bench_largeScale_ablation_and_other_methods_deep-1B.png
       :width: 700
       :alt: deep-96-1B compression evaluation results


.. collapse:: DPR-768-10M

    Results for the DPR-768-10M dataset

    .. image:: ../figs/bench_largeScale_ablation_and_other_methods_dpr-10M.png
       :width: 700
       :alt: DPR-768-10M compression evaluation results

|

.. |copy|   unicode:: U+000A9 .. COPYRIGHT SIGN
.. |reg|   unicode:: U+00AE .. REGISTERED

.. rubric:: Footnotes

.. [#ft1] Performance varies by use, configuration and other factors. Learn more at `www.Intel.com/PerformanceIndex <www.Intel.com/PerformanceIndex/>`_.
       Performance results are based on testing as of dates shown in configurations and may not reflect all publicly
       available updates. No product or component can be absolutely secure. Your costs and results may vary. Intel
       technologies may require enabled hardware, software or service activation. |copy| Intel Corporation.  Intel,
       the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries.  Other names and
       brands may be claimed as the property of others.

.. [#ft3] All experimental results were completed by April 30th 2023.

.. [#ft2] NGT [IwMi18]_ is included in the :ref:`small_scale_benchs` and not in the large scale evaluation because the algorithm designed for
       large-scale datasets (NGT-QBG) achieves  low accuracy saturating at 0.86 recall even for a small 1-million vectors dataset.