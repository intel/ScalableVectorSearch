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
