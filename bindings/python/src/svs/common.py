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

import itertools
import importlib
import struct
import os

import numpy as np
from .loader import library
lib = library()

def np_to_svs(nptype):
    """
    Convert the provided numpy type to the corresponding SVS type enum.
    Throws an unspecified exception if no such conversion is possible.

    Args:
        nptype: The numpy type to obtain the SVS enum for.

    Returns:
        An SVS enum equivalent for the argument.
    """
    # Signed
    if nptype == np.int8:
        return lib.int8
    if nptype == np.int16:
        return lib.int16
    if nptype == np.int32:
        return lib.int32
    if nptype == np.int64:
        return lib.int64
    # Unsigned
    if nptype == np.uint8:
        return lib.uint8
    if nptype == np.uint16:
        return lib.uint16
    if nptype == np.uint32:
        return lib.uint32
    if nptype == np.uint64:
        return lib.uint64
    # Float
    if nptype == np.float16:
        return lib.float16
    if nptype == np.float32:
        return lib.float32
    if nptype == np.float64:
        return lib.float64

    raise Exception(f"Could not convert {nptype} to a svs.DataType enum!");

def read_npy(filename: str):
    """
    Read a file in the `npy` format and return a NumPy array with the results.
    Args:
        filename: The file to read.

    Returns:
        Numpy array with the results.
    """

    X = np.load(filename)
    return np.ascontiguousarray(X)

def read_vecs(filename: str):
    """
    Read a file in the `bvecs/fvecs/ivecs` format and return a NumPy array with the results.

    The data type of the returned array is determined by the file extension with the
    following mapping:

    * `bvecs`: 8-bit unsigned integers.
    * `fvecs`: 32-bit floating point numbers.
    * `ivecs`: 32-bit signed integers.

    Args:
        filename: The file to read.

    Returns:
        Numpy array with the results.
    """

    file_type = filename[-5:]
    if file_type == 'bvecs':
        dtype = np.uint8
        struct_format = 'B'
        n_bytes = 1
        padding = 4
    elif file_type == 'fvecs':
        dtype = np.float32
        struct_format = 'f'
        n_bytes = 4
        padding = 1
    elif file_type == 'ivecs':
        dtype = np.uint32
        struct_format = 'i'
        n_bytes = 4
        padding = 1
    else:
        raise ValueError('Can only open bvecs, fvecs, and ivecs.')

    with open(filename, 'rb') as fin:
        vec_size = struct.unpack('i', fin.read(4))[0]

    X = np.fromfile(filename, dtype=dtype)
    X = X.reshape((-1, vec_size + padding))
    X = X[:, padding:]
    return np.ascontiguousarray(X)

def read_svs(filename: str, dtype = np.float32):
    """
    Read the svs native data file as a numpy array.
    *Note*: As of no, now type checking is performed. Make sure the requested type actually
    matches the contents of the file.

    Args:
        filename: The file to read.
        dtype: The data type of the encoded vectors in the file.

    Result:
        A numpy matrix with the results.
    """
    with open(filename, "rb") as fin:
        # Read through the magic number
        struct.unpack('q', fin.read(8))
        # Read throug the UUID
        _uuid = struct.unpack('q', fin.read(8))
        _uuid = struct.unpack('q', fin.read(8))

        # Get the number of vectors and the dimensionality.
        nvectors = struct.unpack('q', fin.read(8))[0]
        vec_size = struct.unpack('q', fin.read(8))[0]

    header_size = 1024
    X = np.fromfile(filename, dtype = dtype, offset = header_size)
    return np.ascontiguousarray(X.reshape((-1, vec_size)))

def write_vecs(array: np.array, filename: str, skip_check: bool = False):
    """
    Args:
        array: The raw array to save.
        filename: The file where the results will be saved.
        skip_check: Be default, this function will check if the file extension for the vecs
            file is appropriate for the given array (see list below).

            Passing `skip_check = True` overrides this logic and forces creation of the
            file.

    Result:
        The array is saved to the requested file.

    File extention to array element type:

        * fvecs: np.float32
        * hvecs: np.float16
        * ivecs: np.uint32
        * bvecs: np.uint8
    """

    assert(array.ndim == 2)
    dtype = array.dtype.type

    if not skip_check:
        _, extension = os.path.splitext(filename)
        type_extension_map = {
            np.float32: ".fvecs",
            np.float16: ".hvecs",
            np.uint8: ".bvecs",
            np.uint32: ".ivecs",
        }

        if not dtype in type_extension_map:
            raise RuntimeError(f"Unsupported dtype: {dtype}")

        expected_extension = type_extension_map[dtype]
        if extension != expected_extension:
            name = np.dtype(dtype).name
            message = " ".join([
                f"Expected file extension {expected_extension} for data type {name}.",
                f"Instead, got {extension}.",
            ])
            raise RuntimeError(message)


    # Get the vector header as a 4-byte encoded string.
    sz = array.shape[-1]
    sz_bytes = sz.to_bytes(4, "little")
    with open(filename, "wb") as io:
        for i in range(array.shape[0]):
            io.write(sz_bytes)
            io.write(array[i,:].tobytes())

def random_dataset(nvectors: int, ndims: int, dtype = np.float32, seed = None):
    # Use the `RandomState` generator for it's strong backward-compatibility guarentee.
    # I.E., we don't care about performance. We want reproducibility here.
    rng = np.random.RandomState(seed = seed)
    if dtype in (np.float16, np.float32):
        return rng.normal(size = (nvectors, ndims)).astype(dtype)
    elif dtype in (np.uint32, np.uint8, np.int8):
        return rng.randint(100, size = (nvectors, ndims), dtype = dtype)
    else:
        raise Exception(f"Unhandled datatype {dtype}")

def generate_test_dataset(
        nvectors: int,
        nqueries: int,
        ndims: int,
        directory: str,
        data_seed = None,
        query_seed = None,
        num_threads = 1,
        num_neighbors: int = 100,
        distance = lib.DistanceType.L2
        ):
    """
    Generate a sample dataset consisting of the base data, queries, and groundtruth all in
    the standard ``*vecs`` form.

    Args:
        nvectors: The number of base vectors in the generated dataset.
        nqueries: The number of query vectors in the generated dataset.
        ndims: The number of dimensions per vector in the dataset.
        directory: The directory in which to generate the dataset.
        data_seed (optional): The seed to use for random number generation in the dataset.
        query_seed (optional): The seed to use for random number generation for the queries.
        num_threads (optional): Number of threads to use to generate the groundtruth.
        num_neighbors: The number of neighbors to compute for the groundtruth.
        distance (optional): The distance metric to use for groundtruth generation.

    Creates ``directory`` if it didn't already exist. The following files are generated:

    - ``$(directory)/data.fvecs``: The dataset encoded using float32 in as fvecs.
    - ``$(directory)/queries.fvecs``: The queries encoded using float32 as fvecs.
    - ``$(directory)/groundtruth.ivecs``: The computed ``num_neighbors`` nearest
      neighbors of the queries in the dataset with respect to the provided distance.

    """

    if not os.path.isdir(directory):
        os.mkdir(directory)

    print("Generating Data")
    data = random_dataset(nvectors, ndims, dtype = np.float32, seed = data_seed)
    write_vecs(data, os.path.join(directory, "data.fvecs"))

    print("Generating Queries")
    queries = random_dataset(nqueries, ndims, dtype = np.float32, seed = query_seed)
    write_vecs(queries, os.path.join(directory, "queries.fvecs"))

    print("Generating Groundtruth")
    index = lib.Flat(data, distance, num_threads = num_threads)
    I, _ = index.search(queries, num_neighbors)
    write_vecs(I.astype(np.uint32), os.path.join(directory, "groundtruth.ivecs"))


def k_recall_at(gt_idx, result_idx, k: int, at: int):
    if k > at:
        raise ValueError(f'K={k} is higher than @={at}')
    if gt_idx.shape[1] < k:
        raise ValueError(f'Too few ground truth neighbors'
                         f'({gt_idx.shape[1]}) to compute {k}-recall')
    if result_idx.shape[1] < at:
        raise ValueError(f'Too few approximate neighbors'
                         f'({result_idx.shape[1]}) to compute recall@{at}')

    ls_intersection = itertools.starmap(np.intersect1d,
                                        zip(gt_idx[:, :k], result_idx[:, :at]))

    ls_recall = [len(intersect) for intersect in ls_intersection]

    return sum(ls_recall) / (len(ls_recall) * k)
