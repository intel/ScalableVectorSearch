.. _experimental_methodology:

Experimental Setup
******************

.. contents::
   :local:
   :depth: 1

.. _datasets:

Datasets
=========
To cover a wide range of use cases, we evaluate SVS on standard datasets of diverse dimensionalities (:math:`d=25`
to :math:`d=768`), number of elements (:math:`n=10^6` to :math:`n=10^9`), data types and metrics as described in the table below.

+-----------------------------------------------------------------------------------------------------+--------+-------+--------------+-------------------+---------------+-----------------+
| **Dataset**                                                                                         | **d**  | **n** | **Encoding** | **Similarity**    | **n queries** | **Space (GiB)** |
+-----------------------------------------------------------------------------------------------------+--------+-------+--------------+-------------------+---------------+-----------------+
|                                                                                                     |        |       |              |                   |               |                 |
| `gist-960-1M   <http://corpus-texmex.irisa.fr/>`_                                                   | 960    | 1M    | float32      | L2                | 1000          | 3.6             |
+-----------------------------------------------------------------------------------------------------+--------+-------+--------------+-------------------+---------------+-----------------+
| `sift-128-1M   <http://corpus-texmex.irisa.fr/>`_                                                   | 128    | 1M    | float32      | L2                | 10000         | 0.5             |
+-----------------------------------------------------------------------------------------------------+--------+-------+--------------+-------------------+---------------+-----------------+
| `deep-96-10M    <http://sites.skoltech.ru/compvision/noimi/>`_                                      | 96     | 10M   | float32      | cosine similarity | 10000         | 3.6             |
+-----------------------------------------------------------------------------------------------------+--------+-------+--------------+-------------------+---------------+-----------------+
| `glove-50-1.2M   <https://nlp.stanford.edu/projects/glove/>`_                                       | 50     | 1.2M  | float32      | cosine similarity | 10000         | 0.2             |
+-----------------------------------------------------------------------------------------------------+--------+-------+--------------+-------------------+---------------+-----------------+
| `glove-25-1.2M   <https://nlp.stanford.edu/projects/glove/>`_                                       | 25     | 1.2M  | float32      | cosine similarity | 10000         | 0.1             |
+-----------------------------------------------------------------------------------------------------+--------+-------+--------------+-------------------+---------------+-----------------+
| :ref:`DPR-768-10M <DPR_dataset>`                                                                    | 768    | 10M   | float32      | inner product     | 10000         | 28.6            |
+-----------------------------------------------------------------------------------------------------+--------+-------+--------------+-------------------+---------------+-----------------+
| `t2i-200-100M   <https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search>`_ | 200    | 100M  | float32      | inner product     | 10000         | 74.5            |
+-----------------------------------------------------------------------------------------------------+--------+-------+--------------+-------------------+---------------+-----------------+
| `deep-96-100M    <http://sites.skoltech.ru/compvision/noimi/>`_                                     | 96     | 100M  | float32      | cosine similarity | 10000         | 35.8            |
+-----------------------------------------------------------------------------------------------------+--------+-------+--------------+-------------------+---------------+-----------------+
| `deep-96-1B    <http://sites.skoltech.ru/compvision/noimi/>`_                                       | 96     | 1B    | float32      | cosine similarity | 10000         | 257.6           |
+-----------------------------------------------------------------------------------------------------+--------+-------+--------------+-------------------+---------------+-----------------+
|                                                                                                     |        |       |              |                   |               |                 |
| `sift-128-1B <http://corpus-texmex.irisa.fr/>`_                                                     | 128    | 1B    | uint8        | L2                | 10000         | 119.2           |
+-----------------------------------------------------------------------------------------------------+--------+-------+--------------+-------------------+---------------+-----------------+

.. _DPR_dataset:

DPR-768-10M Dataset
--------------------
DPR is a dataset containing 10 million 768-dimensional embeddings generated with the dense passage retriever (DPR)
[KOML20]_ model. Text snippets from the C4 dataset [RSRL20]_ are used to generate: 10 million context DPR embeddings
(base set) and 10000 question DPR embeddings (query set). The code to generate the dataset can be found `here <https://github.com/IntelLabs/DPR-dataset-generator>`_.

Evaluation Metrics
==================
In all benchmarks and experimental results, search accuracy is measured by k-recall at k, defined by
:math:`| S \cap G_t | / k`, where :math:`S` are the ids of the :math:`k` retrieved neighbors and
:math:`G_t` is the ground-truth. We use :math:`k=10` in all experiments.
Search performance is measured by queries per second (QPS).

.. _query_batch_size:

Query Batch Size
================

The size of the query batch, which will depend on the use case, can have a big impact on performance. Therefore,
we evaluate batch sizes: 1 (one query at a time or single query), 128 (typical batch size for deep learning training
and inference) and full batch (determined by the number of queries in the dataset, see :ref:`datasets`).