#!/bin/bash
#
# Copyright (C) 2024-present, Intel Corporation
#
# You can redistribute and/or modify this software under the terms of the
# GNU Affero General Public License version 3.
#
# You should have received a copy of the GNU Affero General Public License
# version 3 along with this software. If not, see
# <https://www.gnu.org/licenses/agpl-3.0.en.html>.
#

# install and source oneMKL
export ONEAPI_INSTALL_DIR="$PREFIX/oneapi"
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/2f3a5785-1c41-4f65-a2f9-ddf9e0db3ea0/l_onemkl_p_2024.1.0.695_offline.sh
bash l_onemkl_p_2024.1.0.695_offline.sh -a -s --eula accept --install-dir $ONEAPI_INSTALL_DIR
rm l_onemkl_p_2024.1.0.695_offline.sh

source $ONEAPI_INSTALL_DIR/setvars.sh

# pysvs building
export CMAKE_ARGS="$CMAKE_ARGS -DPYSVS_MICROARCHS=cascadelake;icelake;sapphirerapids"
$PYTHON -m pip install bindings/python

# clean up oneAPI installation or 
# conda-build will try to `patchelf` its libs with failure
rm -rf $ONEAPI_INSTALL_DIR
