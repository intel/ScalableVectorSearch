#
# Copyright (C) 2023-present, Intel Corporation
#
# You can redistribute and/or modify this software under the terms of the
# GNU Affero General Public License version 3.
#
# You should have received a copy of the GNU Affero General Public License
# version 3 along with this software. If not, see
# <https://www.gnu.org/licenses/agpl-3.0.en.html>.
#

# Test helper for dynamic datasets.
import numpy as np
import pysvs

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
        self.raw_data = pysvs.read_vecs(test_data_vecs)
        self.queries = pysvs.read_vecs(test_queries)
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
        index = pysvs.Flat(sub_dataset, pysvs.DistanceType.L2, self.num_threads)
        I, D = index.search(self.queries, num_neighbors)
        return ids_np[I]

