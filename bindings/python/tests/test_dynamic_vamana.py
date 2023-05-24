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

# unit under test
import pysvs

# stdlib
import unittest

# helpers
from .dynamic import ReferenceDataset

class DynamicVamanaTester(unittest.TestCase):
    """
    Test building, adding, deleting points from the dynamic vamana index.
    """

    def id_check(self, index, ids):
        # Check that every id in `ids` is in the index.
        for this_id in ids:
            self.assertTrue(index.has_id(this_id))

        # Check that every id in the index is in `ids`
        all_ids = index.all_ids()
        for this_id in all_ids:
            self.assertTrue(this_id in ids)

    def recall_check(
            self,
            index: pysvs.DynamicVamana,
            reference: ReferenceDataset,
            num_neighbors: int,
            expected_recall,
            recall_delta,
        ):
        gt = reference.ground_truth(num_neighbors)
        I, D = index.search(reference.queries, num_neighbors)
        recall = pysvs.k_recall_at(gt, I, num_neighbors, num_neighbors)
        print("    Recall: ", recall)
        self.assertTrue(recall < expected_recall + recall_delta)
        self.assertTrue(recall > expected_recall - recall_delta)


    def test_loop(self):
        num_threads = 2
        num_neighbors = 10
        num_tests = 10
        consolidate_every = 2
        delta = 1000

        # Recall can fluctuate up and down.
        # here, we set an expected mid-point for the recall and allow it to wander up and
        # down by a little.
        expected_recall = 0.845
        expected_recall_delta = 0.03

        reference = ReferenceDataset(num_threads = num_threads)
        data, ids = reference.new_ids(5000)

        parameters = pysvs.VamanaBuildParameters(
            graph_max_degree = 64,
            window_size = 128,
            num_threads = num_threads,
            alpha = 1.2,
        )

        index = pysvs.DynamicVamana.build(
            parameters,
            data,
            ids,
            pysvs.DistanceType.L2,
            num_threads,
        )

        self.assertEqual(index.alpha, parameters.alpha)
        self.assertEqual(index.construction_window_size, parameters.window_size)

        # Perform an ID check
        self.id_check(index, reference.ids())

        # Groundtruth Check
        index.search_window_size = 20
        print("Initial")
        self.recall_check(
            index, reference, num_neighbors, expected_recall, expected_recall_delta
        )

        consolidate_count = 0
        for i in range(num_tests):
            (data, ids) = reference.new_ids(delta)
            index.add(data, ids)
            print("Add")
            self.id_check(index, reference.ids())
            self.recall_check(
                index, reference, num_neighbors, expected_recall, expected_recall_delta
            )

            ids = reference.remove_ids(delta)
            index.delete(ids)
            print("Delete")
            self.id_check(index, reference.ids())
            self.recall_check(
                index, reference, num_neighbors, expected_recall, expected_recall_delta
            )

            consolidate_count += 1
            if consolidate_count == consolidate_every:
                index.consolidate().compact(1000)
                self.id_check(index, reference.ids())
                print("Cleanup")
                self.recall_check(
                    index, reference, num_neighbors, expected_recall, expected_recall_delta
                )

