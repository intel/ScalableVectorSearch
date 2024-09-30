# Copyright (C) 2023 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials,
# and your use of them is governed by the express license under which they
# were provided to you ("License"). Unless the License provides otherwise,
# you may not use, modify, copy, publish, distribute, disclose or transmit
# this software or the related documents without Intel's prior written
# permission.
#
# This software and the related documents are provided as is, with no
# express or implied warranties, other than those that are expressly stated
# in the License.

import numpy as np
from typing import Tuple

def _loss(QtQ: np.ndarray, XtX: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    M1 = B @ A.T @ QtQ @ A @ B.T @ XtX
    M2 = QtQ @ A @ B.T @ XtX
    return np.trace(M1) - 2 * np.trace(M2) + np.trace(QtQ @ XtX)


def _gradA(QtQ: np.ndarray, XtX: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return 2 * QtQ @ A @ B.T @ XtX @ B - 2 * QtQ @ XtX @ B


def _gradB(QtQ: np.ndarray, XtX: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return 2 * XtX @ B @ A.T @ QtQ @ A - 2 * XtX @ QtQ @ A


def _frank_wolfe_step(QtQ: np.ndarray, XtX: np.ndarray, A: np.ndarray, \
        B: np.ndarray, grad_fun) -> np.ndarray:
    M = grad_fun(QtQ, XtX, A, B)
    U, _, Vt = np.linalg.svd(-M, full_matrices=False)
    return U @ Vt


def _orthogonality_error(M: np.ndarray) -> np.ndarray:
    d = M.shape[1]
    return np.linalg.norm(M.T @ M - np.eye(d)) ** 2 / d


def compute_leanvec_matrices(X: np.ndarray, Q: np.ndarray, n_components: int,
        n_max_steps: int = 500, rel_tol:float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes query and data matrices for dimensionality reduction using
    the LeanVec Out-Of-Distribution (OOD) learning technique described in

    Tepper, Bhati, Aguerrebere, Hildebrand, Willke:
    "LeanVec: Search your vectors faster by making them fit", 2023,
    (https://arxiv.org/abs/2312.16335)

    Args:
        X: Base dataset array. For faster computation, use up to 100K vectors
        Q: Query array. To avoid overfitting, use different sets to compute matrices and test
        n_components: Desired reduced dimensionality
        n_max_steps: Maximum steps used in the learning algorithm
        rel_tol: Relative tolernace margin in loss before the optimization stops

    Result:
        Data matrix (float32) of shape  `orginal dimensions X n_components`
        Query matrix (float32) of shape  `orginal dimensions X n_components`
    """
    if n_components > X.shape[1]:
        raise ValueError('Too many components')

    QtQ = Q.T @ Q
    XtX = X.T @ X

    A = np.zeros((Q.shape[1], n_components))
    B = np.zeros((X.shape[1], n_components))

    ls_loss = []
    ls_orthogonality_A = []
    ls_orthogonality_B = []

    for it in range(n_max_steps):
        gamma = 1 / (it + 1) ** 0.999
        deltaA = _frank_wolfe_step(QtQ, XtX, A, B, _gradA)
        A = (1 - gamma) * A + gamma * deltaA
        deltaB = _frank_wolfe_step(QtQ, XtX, A, B, _gradB)
        B = (1 - gamma) * B + gamma * deltaB

        loss_val = _loss(QtQ, XtX, A, B)
        ls_loss.append(loss_val)

        ls_orthogonality_A.append(_orthogonality_error(A))
        ls_orthogonality_B.append(_orthogonality_error(B))

        if it > 1 and np.abs(loss_val - ls_loss[-2]) < rel_tol * ls_loss[-2]:
            break

    n_actual_steps = len(ls_loss)

    return B.astype(np.float32), A.astype(np.float32)
