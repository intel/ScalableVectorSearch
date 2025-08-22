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

C++ Library
===========

The SVS core library is mainly a header-only library with some future additional utilities potentially being moved into a small shared-library.
The reason for this design decision is to make maximum use of the compiler when we can (using features like compile-time vector dimensionality, statically resolved function calls, etc.) and to allow for flexible internal interfaces.
While this has the potential to increase binary size over an object-oriented approach due to extra template instantiation, the goal to tackle this is through careful type erasure are critical interfaces.
Using this approach, we can measure the performance impact of type erasure (i.e., dynamic dispatch) to maintain high performance.

The high-level C++ API is described and documented in the following sections.

