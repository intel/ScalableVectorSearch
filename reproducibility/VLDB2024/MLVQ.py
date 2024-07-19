import svs
import numpy as np
import os
import multiprocessing as multip
import faiss


class MLVQ:
    def __init__(self, M, B1, B2, X=None, centroids=None, cluster_assignments=None, padding=32):
        self.M = M
        self.B1 = B1
        self.B2 = B2
        self.padding = padding
        self.strategy = svs.LVQStrategy.Turbo if self.B1 == 4 and self.B2 == 8 \
            else svs.LVQStrategy.Sequential

        if centroids is None:
            assert X is not None, 'Need to provide X to compute the centroids.'
            self.centroids, self.cluster_assignments = self.generate_clusters(X)
        else:
            self.centroids = centroids
            self.cluster_assignments = cluster_assignments

    def generate_clusters(self, X, niter=50, verbose=False, seed=0, useGPU=False, max_points_per_centroid=256):
        d = X.shape[1]
        kmeans = faiss.Kmeans(d, self.M, niter=niter, verbose=verbose, seed=seed, gpu=useGPU,
                                      max_points_per_centroid=max_points_per_centroid)
        kmeans.train(X)
        _, I = kmeans.index.search(X, 1)
        return kmeans.centroids, I.astype('uint64')


class ExperimentalSetup:
    def __init__(self, centroids_fname, clust_assignment_fname, data_compr_fname, compress_datadir, num_threads,
                 configdir, graphdir, datadir, dist_type, graph_max_degree, alpha, num_neighbors):
        self.centroids_fname = centroids_fname
        self.clust_assignment_fname = clust_assignment_fname
        self.data_compr_fname = data_compr_fname
        self.compress_datadir = compress_datadir
        self.num_threads = num_threads
        self.configdir = configdir
        self.graphdir = graphdir
        self.datadir = datadir
        self.dist_type = dist_type
        self.graph_build_parameters = svs_build_parameters(graph_max_degree, alpha)
        self.num_neighbors = num_neighbors


def svs_build_parameters(graph_max_degree, alpha, window_size=200, max_candidate_pool_size = 750):
    buildParams = svs.VamanaBuildParameters(
        graph_max_degree=graph_max_degree,
        window_size=window_size,
        alpha=alpha,
        max_candidate_pool_size=max_candidate_pool_size,
        prune_to=(graph_max_degree-4)
    )
    return buildParams

def update_svs_LVQ_dataset_file(X, cluster_assignments, centroids, mlvq, expSetup, ids_sort=None):
    dims = X.shape[1]

    # Check no NaNs are included
    if ids_sort is not None:
        if np.isnan(ids_sort).any() or np.isinf(ids_sort).any():
            raise BaseException('update_svs_LVQ_dataset_file: NaN found in ids_sort')
    if np.isnan(X).any() or np.isinf(X).any():
        raise BaseException('update_svs_LVQ_dataset_file: NaN found in X')
    if np.isnan(cluster_assignments).any() or np.isinf(cluster_assignments).any():
        raise BaseException('update_svs_LVQ_dataset_file: NaN found in cluster_assignments')
    if np.isnan(centroids).any() or np.isinf(centroids).any():
        raise BaseException('update_svs_LVQ_dataset_file: NaN found in centroids')

    if ids_sort is not None:
        print('Saving auxiliary files with ids reasignment...')
        svs.write_vecs(X.astype('float32')[ids_sort],
                         expSetup.data_compr_fname)  # These points should be in the correct order!
        save_assignments(cluster_assignments[ids_sort], expSetup.clust_assignment_fname)
    else:
        print('Saving auxiliary files without ids reasignment...')
        svs.write_vecs(X.astype('float32'),
                         expSetup.data_compr_fname)  # These points should be in the correct order!
        save_assignments(cluster_assignments, expSetup.clust_assignment_fname)

    svs.write_vecs(centroids.astype('float32'), expSetup.centroids_fname)
    print('SVS compression...')
    if not os.path.exists(expSetup.compress_datadir):
            os.makedirs(expSetup.compress_datadir)

    svs.reproducibility.compress(svs.LVQLoader("", primary=mlvq.B1, residual=mlvq.B2),
                   expSetup.data_compr_fname,
                   expSetup.centroids_fname, expSetup.clust_assignment_fname,
                   expSetup.compress_datadir, expSetup.num_threads)
    print('Finished SVS compression...')

def save_assignments(assignments, assignment_fname):
    assignments.astype('uint64').tofile(assignment_fname)

def compressed_vector_loader(dataset_path, B1, B2, padding, strategy, dims):
    return svs.LVQLoader(dataset_path, primary=B1, residual=B2, strategy=strategy, padding=padding, dims=dims)

def ids_for_sorting_dict(id):
    return ids_map_dict[id]

def ids_for_sorting(current_ids, ids_mapping):
    n = current_ids.shape[0]
    if ids_mapping.shape[0] != n:
        raise BaseException(f'Indices size mismatch: lvq_idx has {n} and SVS ids are {ids_mapping.shape[0]}')

    print('Building dict for ids mapping')
    global ids_map_dict
    ids_map_dict = {current_ids[i]: i for i in range(n)}
    print('Finished building dict for ids mapping')

    print(f'Starting pool call with {multip.cpu_count()} cores')
    with multip.Pool(processes=multip.cpu_count()) as p:
        ids_aux_pool = p.map(ids_for_sorting_dict, [ids_mapping[i, 0] for i in range(n)])
    print('Finished pool call')
    ids_aux = np.concatenate([ids_mapping, np.expand_dims(np.asarray(ids_aux_pool, dtype=np.float32), axis=1)], axis=1)
    ids_aux = ids_aux[ids_aux[:, 1].argsort()]

    return ids_aux[:, 2].astype('int')

def internal_external_id_mapping(configdir, current_ids):
    ids_file = f'{configdir}/id_translation_0.binary'
    print('Loading ids for SVS mapping')
    n_nodes = current_ids.shape[0]
    ids_mapping = load_svs_internal_external_id_mapping(ids_file, n_nodes)
    print('Sorting ids for SVS mapping')
    ids_sort = ids_for_sorting(current_ids, ids_mapping)
    print('Finished sorting ids for SVS mapping')
    return ids_sort

def load_svs_internal_external_id_mapping(ids_file, n_nodes):
    size_t_size = int(8)
    size_idx = int(4)
    data = np.memmap(ids_file, dtype='uint8', mode='r')
    ids_mapping = np.zeros([n_nodes, 2])

    print('Starting to load nodes for ids mapping...')
    init = int(0)
    for i in range(n_nodes):
        ids_mapping[i, 0] = int(data[init:init + size_t_size].view('uint64')[0])  # External id
        init += int(size_t_size)
        ids_mapping[i, 1] = int(data[init:init + size_idx].view('uint32')[0])  # Internal id
        init += int(size_idx)
    print('Finished loading nodes for ids mapping.')

    return ids_mapping

def update_and_reload_compressed_graph(X, cluster_assignments, centroids, mlvq, expSetup, vector_ids=None):
    # compress_datadir is the folder where the M-LVQ dataset will be created.
    # This dataset will be used to reload the graph.

    dims = X.shape[1]

    if vector_ids is not None:
        # This is for dynamic indexing only, when the external vector ids (vector_ids) may not match the internal SVS ids
        ids_sort = internal_external_id_mapping(expSetup.configdir, vector_ids)
    else:
        ids_sort = None
    update_svs_LVQ_dataset_file(X, cluster_assignments, centroids, mlvq, expSetup, ids_sort=ids_sort)

    print(f'Loading graph: {expSetup.configdir}, {expSetup.graphdir}, {expSetup.compress_datadir}')
    index = svs.DynamicVamana(
        expSetup.configdir,
        svs.GraphLoader(expSetup.graphdir),
        compressed_vector_loader(expSetup.compress_datadir, mlvq.B1, mlvq.B2, mlvq.padding, mlvq.strategy,
                                 dims),
        expSetup.dist_type,
        num_threads=expSetup.num_threads,
    )
    return index

def build_and_save(X, expSetup):
    index = svs.DynamicVamana.build(
            expSetup.graph_build_parameters,
            X.astype('float32'),
            np.arange(X.shape[0]).astype('uint64'),
            expSetup.dist_type,
            expSetup.num_threads,
    )
    index.save(expSetup.configdir, expSetup.graphdir, expSetup.datadir)

def main():
    base_dir = '/home/user/research/ann/experiments/papers/vldb2024/code/'
    fname_dataset = '/home/user/research/datasets/deep/deep100k.fvecs'
    fname_queries = '/home/user/research/datasets/deep/deep1b_queries.fvecs'
    fname_gtruth = '/home/user/research/datasets/deep/ground_truth/deep100k_groundtruth.ivecs'

    # Parameters
    M = 5
    B1 = 4
    B2 = 8
    num_threads = 72
    num_neighbors = 10
    dist_type = svs.DistanceType.MIP
    graph_max_degree = 64
    alpha = 1.2 if dist_type == svs.DistanceType.L2 else 0.95

    # Auxiliary files and folder
    centroids_fname = f'{base_dir}/indices/centroids.fvecs'
    clust_assignment_fname = f'{base_dir}/indices/cluster_assignments.bin'
    data_compr_fname = f'{base_dir}/indices/compr_data.fvecs'
    compress_datadir = f'{base_dir}/indices/compr_dir/'
    configdir = f'{base_dir}/indices/svsConfig_dir/'
    graphdir = f'{base_dir}/indices/svsGraph_dir/'
    datadir = f'{base_dir}/indices/svsData_dir/'

    # Load experimental auxiliary variables
    expSetup = ExperimentalSetup(centroids_fname, clust_assignment_fname, data_compr_fname, compress_datadir,
                                 num_threads, configdir, graphdir, datadir, dist_type, graph_max_degree, alpha,
                                 num_neighbors)

    # Load the dataset
    X = svs.read_vecs(fname_dataset)
    Q = svs.read_vecs(fname_queries)
    gtruth = svs.read_vecs(fname_gtruth)

    # Initialize M-LVQ
    mlvq = MLVQ(M, B1, B2, X=X)

    # Build and save the dynamic index
    build_and_save(X, expSetup)

    # Re-load the index ready to run the search with M-LVQ compression
    index = update_and_reload_compressed_graph(X, mlvq.cluster_assignments, mlvq.centroids, mlvq, expSetup)

    # Run the search
    index.num_threads = expSetup.num_threads
    index.search_window_size = 20
    I, _ = index.search(Q, expSetup.num_neighbors)
    recall = svs.k_recall_at(gtruth, I, expSetup.num_neighbors, expSetup.num_neighbors)

    print(f'Recall: {recall}')


if __name__ == '__main__':
    main()

