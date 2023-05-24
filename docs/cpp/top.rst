C++ Library
===========

The SVS core library is mainly a header-only library with some future additional utilities potentially being moved into a small shared-library.
The reason for this design decision is to make maximum use of the compiler when we can (using features like compile-time vector dimensionality, statically resolved function calls, etc.) and to allow for flexible internal interfaces.
While this has the potential to increase binary size over an object-oriented approach due to extra template instantiation, the goal to tackle this is through careful type erasure are critical interfaces.
Using this approach, we can measure the performance impact of type erasure (i.e., dynamic dispatch) to maintain high performance.

The high-level C++ API is described and documented in the following sections.

