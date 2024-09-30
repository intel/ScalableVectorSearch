.. Copyright (C) 2024 Intel Corporation
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

* A C++20 capable compiler:

  * GCC >= 11.0
  * Clang >= 13.0

* `OneMKL <https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html>`_

  * Make sure you set the OneMKL environment variables, e.g., ``source /opt/intel/oneapi/setvars.sh``.

  .. collapse:: Click to expand installation details

      `OneMKL <https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html>`_ can be installed as part of the
      `Intel oneAPI Base Toolkit
      <https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html#gs.8u2swh>`_
      by following one of the methods indicated in the `oneAPI docs
      <https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2024-1/installation.html>`_
      .

      For example, the following commands show how to install the OneMKL component of the Intel oneAPI Base Toolkit on a Linux
      system using the `offline installer <https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html>`_:

      .. code-block:: sh

          wget [link to the offline installer]
          sudo sh [downloaded installer script] -a --components intel.oneapi.lin.mkl.devel --action install --eula accept -s
          source /opt/intel/oneapi/setvars.sh
