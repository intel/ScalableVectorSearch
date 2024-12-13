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

# unit under test
import svs

# stdlib
import unittest
import os
from tempfile import TemporaryDirectory

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
            index: svs.DynamicVamana,
            reference: ReferenceDataset,
            num_neighbors: int,
            expected_recall,
            recall_delta,
        ):
        gt = reference.ground_truth(num_neighbors)
        I, D = index.search(reference.queries, num_neighbors)
        recall = svs.k_recall_at(gt, I, num_neighbors, num_neighbors)
        print("    Recall: ", recall)
        self.assertTrue(recall < expected_recall + recall_delta)
        self.assertTrue(recall > expected_recall - recall_delta)

        # Make sure saving and reloading work.
        with TemporaryDirectory() as tempdir:
            configdir = os.path.join(tempdir, "config")
            graphdir = os.path.join(tempdir, "graph")
            datadir = os.path.join(tempdir, "data")
            index.save(configdir, graphdir, datadir);

            reloaded = svs.DynamicVamana(
                configdir,
                svs.GraphLoader(graphdir),
                svs.VectorDataLoader(datadir, svs.DataType.float32),
                svs.DistanceType.L2,
                num_threads = 2,
            )

            self.assertEqual(index.search_window_size, reloaded.search_window_size)
            self.assertEqual(index.alpha, reloaded.alpha)
            self.assertEqual(index.construction_window_size, reloaded.construction_window_size)

            ### Get recall with the saved search-window-size and other search parameters.
            I, D = reloaded.search(reference.queries, num_neighbors)
            reloaded_recall = svs.k_recall_at(gt, I, num_neighbors, num_neighbors)

            # Because saving triggers graph compaction, we can't guarentee that the reloaded
            # recall is the same as the original index.
            print(f"    Reloaded Recall: {reloaded_recall}")
            self.assertTrue(reloaded_recall < expected_recall + recall_delta)
            self.assertTrue(reloaded_recall > expected_recall - recall_delta)

            # Make sure that search still works even when we set the search parameters
            # to default values.
            reloaded.search_parameters = svs.VamanaSearchParameters()
            I, D = reloaded.search(reference.queries, num_neighbors)

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

        parameters = svs.VamanaBuildParameters(
            graph_max_degree = 64,
            window_size = 128,
            alpha = 1.2,
        )

        index = svs.DynamicVamana.build(
            parameters,
            data,
            ids,
            svs.DistanceType.L2,
            num_threads = num_threads,
        )

        print(f"Testing {index.experimental_backend_string}")

        index.search_window_size = 10
        self.assertEqual(index.search_window_size, 10)
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
                consolidate_count = 0

