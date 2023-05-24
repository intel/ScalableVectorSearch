# Scalable Vector Search

**Scalable Vector Search (SVS)** is a performance library for vector [similarity search](https://en.wikipedia.org/wiki/Similarity_search).
SVS enables application and framework developers using similarity search to unleash its performance on Intel(R) XPUs.
SVS provides vector similarity search:
* on **billions** of **high-dimensional** vectors,
* at **high accuracy**
* and **state-of-the-art speed**,
* while enabling the use of **less memory** than its alternatives.

As an example, for *one billion* 96-dimensional vectors ([Deep-1B](http://sites.skoltech.ru/compvision/noimi/)),
SVS outcompetes the alternatives [[ABHT23]](#1) as follows[^1]:

<p align="center">
<img src="./docs/figs/SVS_performance_memoryfootprint.png" height=80% width=80%>
</p>

All in all, SVS achieves:

| SVS configuration | Max throughput advantage | Max memory savings |
|-------------------------|:------------------------:|:------------------:|
| low-memory (R=32) | 20.7x | 3x |
| high-throughput (R=126) | 5.8x | 1.4x |

See [Benchmarks]() for more details.

SVS is written in C++, with complete Python bindings.

## Key Features

SVS supports the following:
* Similarity functions: Euclidean distance, inner product, cosine similarity.
* Vectors with individual values encoded as: float32, float16, uint8, int8.
* Vector compression (including Locally-adaptive Vector Quantization [[1]](#1))
* Optimizations for Intel &reg; Xeon &reg; processors:
  - 2nd generation (Cascade Lake)
  - 3rd generation (Ice Lake)
  - 4th generation (Sapphire Rapids)

See [Roadmap]() for upcoming features.

## Documentation

[Our documentation]() provides a reference of the library API, as well as several guides, tutorials,
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
git clone https://github.com/intel-sandbox/ai.similarity-search.gss
cd ai.similarity-search.gss

# Install pysvs using pip
CC=gcc-11 CXX=g++-11 pip install bindings/python
```

To uninstall, simply run
```
pip uninstall pysvs
```

For more advanced building options see [LINK].

## Usage Example
See the example in [LINK], or for more details [LINK to Getting Started].

## Library Philosophy

The SVS core library is mainly a header-only library with some future additional utilities potentially being moved into a small shared-library.
The reason for this design decision is to make maximum use of the compiler when we can (using features like compile-time vector dimensionality, statically resolved function calls, etc.) and to allow for flexible internal interfaces.
While this has the potential to increase binary size over an object-oriented approach due to extra template instantiation, the goal to tackle this is through careful type erasure are critical interfaces.
Using this approach, we can measure the performance impact of type erasure (i.e., dynamic dispatch) to ensure we don't lose performance.

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
We would love to accept your contributions to this project. See [How to Contribute](contributing/CONTRIBUTING.md) for
more details.

## Legal
Refer to the [LICENSE file](LICENSE.md) for details.

[^1]: Performance varies by use, configuration and other factors. Learn more at [www.Intel.com/PerformanceIndex](http://www.Intel.com/PerformanceIndex/).
Performance results are based on testing as of dates shown in configurations and may not reflect all publicly
available updates. No product or component can be absolutely secure. Your costs and results may vary. Intel
technologies may require enabled hardware, software or service activation. &copy; Intel Corporation.  Intel,
the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries.  Other names and
brands may be claimed as the property of others.
