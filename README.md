# Scalable Vector Search

**Scalable Vector Search (SVS)** is a performance library for vector [similarity search](https://en.wikipedia.org/wiki/Similarity_search).
Thanks to the use of Locally-adaptive Vector Quantization [[ABHT23]](#1) and its highly optimized indexing and search algorithms,
SVS provides vector similarity search:
* on **billions** of **high-dimensional** vectors,
* at **high accuracy**
* and **state-of-the-art speed**,
* while enabling the use of **less memory** than its alternatives.

This enables application and framework developers using similarity search to unleash its performance on Intel &reg; Xeon CPUs (2nd generation and newer).

SVS offers a fully-featured and yet simple Python API, compatible with most standard libraries.
SVS is written in C++ to facilitate its integration into performance-critical applications.

## Performance

SVS provides state-of-the-art performance and accuracy [[ABHT23]](#1) for billion-scale similarity search on
[standard benchmarks](https://intellabs.github.io/ScalableVectorSearch/benchs/index.html).

For example, for the standard billion-scale [Deep-1B](http://sites.skoltech.ru/compvision/noimi/) dataset,
different configurations of SVS yield significantly increased performance (measured in queries per second, QPS) with a smaller memory footprint (horizontal axis) than the alternatives[^1]:

<p align="center">
<img src="./docs/figs/SVS_performance_memoryfootprint.png" height=80% width=80%>
</p>

SVS is primarily optimized for large-scale similarity search but it still offers [state-of-the-art performance
at million-scale](https://jubilant-adventure-vr8r2zw.pages.github.io/benchs/small_scale_benchs.html).

## Key Features

SVS supports:
* Similarity functions: Euclidean distance, inner product, cosine similarity.
* Vectors with individual values encoded as: float32, float16, uint8, int8.
* Vector compression (including Locally-adaptive Vector Quantization [[ABHT23]](#1))
* Optimizations for Intel &reg; Xeon &reg; processors:
  - 2nd generation (Cascade Lake)
  - 3rd generation (Ice Lake)
  - 4th generation (Sapphire Rapids)

See [Roadmap](https://intellabs.github.io/ScalableVectorSearch/roadmap.html) for upcoming features.

## Usage Example
Follow a step by step example at [Getting Started](https://intellabs.github.io/ScalableVectorSearch/start.html), or directly access the [entire example code](https://intellabs.github.io/ScalableVectorSearch/start.html#entire-example).

## Documentation

[Our documentation](https://intellabs.github.io/ScalableVectorSearch/index.html) provides a reference of the library API, as well as several guides, tutorials,
and benchmark comparisons.

## Installation

### Prerequisites

* A C++20 capable compiler:

    * GCC >= 11.0
    * Clang >= 13.0

To install the Python module you'll also need:

* Python >= 3.7
* A working internet connection

### Python build

To build and install the Python module, pysvs, clone the repo and run the following pip install command.

```
# Clone the repository
git clone https://github.com/IntelLabs/ScalableVectorSearch.git
cd ScalableVectorSearch

# Install pysvs using pip
CC=gcc-11 CXX=g++-11 pip install bindings/python
```

To uninstall, simply run
```
pip uninstall pysvs
```

For more advanced building options see [Advanced Library Building](https://intellabs.github.io/ScalableVectorSearch/advanced/build.html).

## References
Reference to cite when you use SVS in a research paper:

```
@misc{aguerrebere2023similarity,
title={Similarity search in the blink of an eye with compressed indices},
author={Cecilia Aguerrebere and Ishwar Bhati and Mark Hildebrand and Mariano Tepper and Ted Willke},
year={2023},
eprint={2304.04759},
archivePrefix={arXiv},
primaryClass={cs.LG}
}
```

<a id="1">[ABHT23]</a>
Aguerrebere, C.; Bhati I.; Hildebrand M.; Tepper M.; Willke T.:Similarity search in the blink of an eye with compressed
indices. In: arXiv preprint [arXiv:2304.04759](https://arxiv.org/abs/2304.04759) (2023)

## How to Contribute
We welcome your contributions to this project. See [How to Contribute](contributing/CONTRIBUTING.md) for
more details.

## Legal
Refer to the [LICENSE file](LICENSE) for details.

[^1]: Performance varies by use, configuration and other factors. Learn more at [www.Intel.com/PerformanceIndex](http://www.Intel.com/PerformanceIndex/).
Performance results are based on testing as of dates shown in configurations and may not reflect all publicly
available updates. No product or component can be absolutely secure. Your costs and results may vary. Intel
technologies may require enabled hardware, software or service activation. &copy; Intel Corporation.  Intel,
the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries.  Other names and
brands may be claimed as the property of others.
