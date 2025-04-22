# Copyright 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from typing import Tuple


def compute_leanvec_matrices(X: np.ndarray, Q: np.ndarray, n_components: int,
        n_max_steps: int = 500, rel_tol:float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    A = np.zeros((Q.shape[1], n_components))
    B = np.zeros((X.shape[1], n_components))

    return B.astype(np.float32), A.astype(np.float32)
