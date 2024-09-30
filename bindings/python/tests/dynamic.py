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

# Test helper for dynamic datasets.
import numpy as np
import svs

from .common import test_data_vecs, test_queries

class ReferenceDataset:
    """
    Members

        raw_data: The raw vector data.
        all_ids: All the IDs in the dataset.
        current_ids: The IDs currently in the dataset.
    """

    def __init__(self, num_threads = 2):
        """
        Arguments
            num_threads: The number of threads to use for groundtruth generation.
        """
        self.raw_data = svs.read_vecs(test_data_vecs)
        self.queries = svs.read_vecs(test_queries)
        self.all_ids = np.arange(self.raw_data.shape[0])
        self.current_ids = set()
        self.num_threads = num_threads

    def new_ids(self, n: int):
        np.random.shuffle(self.all_ids)
        ids = []
        for i in self.all_ids:
            if i in self.current_ids:
                continue

            ids.append(i)
            self.current_ids.add(i)
            if len(ids) == n:
                break

        ids_np = np.array(ids, dtype = np.uint64)
        data = self.raw_data[ids_np, :]
        return (data, ids_np)

    def remove_ids(self, n: int):
        np.random.shuffle(self.all_ids)
        ids = []
        for i in self.all_ids:
            if not i in self.current_ids:
                continue

            ids.append(i)
            self.current_ids.remove(i)
            if len(ids) == n:
                break

        return np.array(ids, dtype = np.uint64)

    def ids(self):
        return self.current_ids

    def ground_truth(self, num_neighbors: int):
        # Gather the dataset into a contiguous chunck to pass to the ground truth
        # calculation.
        ids_np = np.array(list(self.current_ids), dtype = np.uint64)
        sub_dataset = self.raw_data[ids_np, :]

        # Create the flat index.
        index = svs.Flat(sub_dataset, svs.DistanceType.L2, self.num_threads)
        I, D = index.search(self.queries, num_neighbors)
        return ids_np[I]

