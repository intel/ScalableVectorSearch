# Intel Similarity Search file format

This document describes the file format(s) used in the library.

The library needs to store two main data structures
- The vector data.
- The index, used to search such vector data.

The IO system needs to support various types of vectors and indices.

# Vector data

Vector data is saved in multiple files:
1. a config file.
    - This config file describes everything needed to correctly load the vectors in the correct format.
    - This file is encoded using TOML.
    - The file name must have the format `[my_file_name].config`
2. optionally, a binary file
   - This file contains the mapping between internal and external IDs.
   - It is encoded as ???
   - The file name must have the format `[my_file_name].svs-id`
3. a list of binary files.
    - The format allows to separate the data in different subsets,
   supporting different encoding schemes, such as
   [multi-level cascade](#multi-level-cascaded-encoding) and
   [blocked](#blocked-encoding) encodings.
    - Each subset is encoded in a separate file.
    - Each file is encoded in the V1 binary file format described [here](#binary-file-format)
    - The file name must have the format `[my_file_name].svs-vec[level_number]`

### Multi-level cascaded encoding
The format supports multi-level cascaded encoding.
By this, we mean that (the symbol `~=` is used to imply a lossy encoding):
- for a single-level cascade each vector `v` is encoded as
  `v ~= v_1`,
  where `v_1` is a lossy or lossless encoding of `v`.
- for a two-level cascade each vector `v` is encoded as
  `v ~= v_1 + v_2`,
  where
   - `v_1` is a lossy encoding of `v`,
   - `v_2` is a (potentially lossy) encoding of `v - v_1`.
- more levels are defined recursively

### Blocked encoding
The format supports assembling the data from multiple blocks, each one in a different file.

These blocks could represent:
- Different data subsets to be concatenated.
- Different hierarchy levels in a hierarchical collection of vectors.

## Config File

The config file begins with a header tagged `[format]`.

| Field   | Allowed values | Description                                                                                                                                            |
|---------|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| version | integer (> 0)  | The version number to support changes of the remainder of the config file or of the binary file in the future. It is a positive integer starting at 1. |
| type    | "vector"       | Allows to check whether this config file represents vector data.                                                                                       |


Example:
```
[format]
version = 1
type = "vector"
```

---

Next, a section tagged `[external_id]` that describes the external IDs of the vectors.

| Field            | Allowed values       | Description                                                                                                                                                                  |
|------------------|----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| present          | true / false         | The presence of external IDs. If external IDs match the order of the vectors in the file, it should be set ot false. If different external IDs are used, the mark it as true |
| format           | "uint64" / "uint32"  | The format of the external IDs                                                                                                                                               |
| binary_file_uuid | standard UUID string | The UUID of the binary file storing the mapping from internal to external IDs                                                                                                |

Example:
```
[external_id]
present = false
format = "uint32"
binary_file_uuid = "9f123207-5fda-4bd6-8eb8-fb9d8e6890f"
```

---

Next, a TOML array of tables tagged `[[encoding]]`, each table describing a subset of the vectors.

| Field            | Allowed values       | Description                                                                                                                                         |
|------------------|----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| uuid             | standard UUID string | The UUID corresponding to the vector data being encoded. The purpose of having this field in each subset is to check the integrity of their origin. |
| n_elements       | integer (>0)         | The number of elements in the subset                                                                                                                |
| format           | "uint64" / "uint32"  | The format of the external IDs                                                                                                                      |
| binary_file_uuid | standard UUID string | The UUID of the binary file storing the subset.                                                                                                     |

The field `format` supports the following values:
- Non-quantized formats
    - "FP16" / "FP32"
    - "uint8" / "int8" / "uint32" / "int32"
- What to do for quantization?

Example:
```
[[encoding]]
uuid = "9e2b41f2-d924-479a-aff0-28ea1a0a5fdf"
n_elements = 1_000_000
format = "FP32"
position = 1
binary_file_uuid = "0c320d8b-2469-46a2-88a3-a0d4884e5cde"
```

---

The table for each subset contains a subtable, tagged `[encoding.params]`, encoding the specific parameters associated with each format (defined in the previous section).
This file format specification does not impose any constraints on the internal format used to serialize the internal parameters of each index, as long as it adheres to TOML.

We now provide a few illustrative examples.

If `format` is a non-quantized type
```
[encoding.params]
n_dimensions = 100
```

If `format` is a quantized type with global constants
```
[encoding.params]
n_dimensions = 10
n_bits = 4  # number of bits per dimension
mean = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # dataset mean
global_min = -0.5  # minimum dataset value after removing the mean
global_max = 2.0  # maximum dataset value after removing the mean
```

If `format` is a quantized type with local constants
```
[encoding.params]
n_dimensions = 10
n_bits = 4  # number of bits per dimension
mean = [1, 2, 3, 4, 5, 6, 5, 4, 1, 2]  # dataset mean
```


# Indices

Vector data is saved in multiple files:
1. a config file.
    - This config file describes everything needed to correctly load the vectors in the correct format.
    - This file is encoded using TOML.
    - The file name must have the format `[my_file_name].config`
3. a binary file.
    - It is encoded in the V1 binary file format described [here](#binary-file-format)
    - The file name must have the format `[my_file_name].svs-idx`

## Config file

The config file begins with a header tagged `[format]`.

| Field   | Allowed values | Description                                                                                                                                            |
|---------|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| version | integer (> 0)  | The version number to support changes of the remainder of the config file or of the binary file in the future. It is a positive integer starting at 1. |
| type    | "vamana"       | Allows to infer the index represented by this this config file.                                                                                        |


Example:
```
[format]
version = 1
type = "vamana"
```

---

Next, a section tagged `[index]`, encoding the parameters that are common to every type of index.

| Field            | Allowed values                                        | Description                                                                                                                                              |
|------------------|-------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| uuid             | standard UUID string                                  | The UUID corresponding to the vector data being encoded. The purpose of having this field is to check the compatibility with the vector data being used. |
| n_elements       | integer (>0)                                          | The number of elements in the index                                                                                                                      |
| internal_id      | "uint32" or "uint64"                                  | The UUID of the binary file storing the mapping from internal to external IDs                                                                            |
| distance         | "squared L2", "cosine similarity", or "inner product" | The distance that should be used with this index                                                                                                         |
| binary_file_uuid | standard UUID string                                  | The UUID of the binary file storing the index.                                                                                                           |


Example:
```
[index]
uuid = "9e2b41f2-d924-479a-aff0-28ea1a0a5fdf"
n_elements = 1_000_000
internal_id = "uint32"
distance = "squared L2"
binary_file_uid = "38fbe63f-fd1c-48d0-adf7-9e4cc8c69717"
```

---

Next, a section tagged `[specific]`, encoding the specific parameters associated with each index.
In V1, only Vamana indices are supported.

| Field                      | Allowed values | Description                                                      |
|----------------------------|----------------|------------------------------------------------------------------|
| entry_point                | integer (>=0)  | The internal ID of the entry point used for searching the graph. |
| max_out_degree             | integer (>0)   | The maximum out degree of the graph.                             |
| alpha                      | float (>0)     | The value of alpha used in the second round of graph updates.    |
| max_candidates             | integer (>0)   | The maximum number of candidates for out-edge pruning.           |
| construction_window_size   | integer (>0)   | The search window size used during graph construction.           |
| default_search_window_size | integer (>0)   | The default search window size.                                  |
| visited_set                | true / false   | Whether to use a visited set to skip distance computations.      |


Example:
```
[specific]
entry_point = 1
max_out_degree = 128
alpha = 1.2
max_candidates = 750
construction_window_size = 100
default_search_window_size = 80
visited_set = false
```

# Binary file format

This file encodes a 2D array.

The header occupies 1024 bytes and contains the following fielsd in this order
1. The magic number for the file encoding.
2. The file UUID encoded with 128 bits as eight uint16 values.
3. The number of rows encoded with 64 bits as an uint64.
4. The number of columns encoded with 64 bits as an uint64.

The 2D array is encoded in row-major format.
