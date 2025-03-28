/*
 * Copyright 2023 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/*****************************************************
 * I/O functions for fvecs, ivecs and xVecs
 *****************************************************/

#include <random>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

int fvec_fwrite(FILE* fo, const float* v, int d) {
    int ret;
    ret = fwrite(&d, sizeof(int), 1, fo);
    if (ret != 1) {
        perror("fvec_fwrite: write error 1");
        return -1;
    }
    ret = fwrite(v, sizeof(float), d, fo);
    if (ret != d) {
        perror("fvec_fwrite: write error 2");
        return -1;
    }
    return 0;
}

int fvecs_write(const char* fname, int d, int n, const float* vf) {
    FILE* fo = fopen(fname, "w");
    if (!fo) {
        perror("fvecs_write: cannot open file");
        return -1;
    }

    int i;
    /* write down the vectors as fvecs */
    for (i = 0; i < n; i++) {
        if (fvec_fwrite(fo, vf + i * d, d) < 0)
            return -1;
    }
    fclose(fo);
    return n;
}

int ivec_iwrite(FILE* fo, const int* v, int d) {
    int ret;
    ret = fwrite(&d, sizeof(int), 1, fo);
    if (ret != 1) {
        perror("fvec_fwrite: write error 1");
        return -1;
    }
    ret = fwrite(v, sizeof(float), d, fo);
    if (ret != d) {
        perror("fvec_fwrite: write error 2");
        return -1;
    }
    return 0;
}

int ivecs_write(const char* fname, int d, int n, const int* vf) {
    FILE* fo = fopen(fname, "w");
    if (!fo) {
        perror("fvecs_write: cannot open file");
        return -1;
    }

    int i;
    /* write down the vectors as fvecs */
    for (i = 0; i < n; i++) {
        if (ivec_iwrite(fo, vf + i * d, d) < 0)
            return -1;
    }
    fclose(fo);
    return n;
}

void generate_random_data(size_t data_dim, size_t dataset_size, size_t query_size) {
    float dataset_std = 1.0f, query_std = 0.1f;

    srand(100);
    std::default_random_engine generator;
    std::normal_distribution<float> dataset_dist(0.0f, dataset_std);
    std::normal_distribution<float> query_dist(0.0f, query_std);
    std::uniform_int_distribution<> uni_dist(0, dataset_size - 1);

    std::vector<float> dataset(dataset_size * data_dim);
    for (size_t i = 0; i < dataset.size(); ++i) {
        dataset[i] = dataset_dist(generator);
    }

    std::vector<float> queries(query_size * data_dim);
    std::vector<int> gt(query_size);
    for (size_t i = 0; i < query_size; ++i) {
        int e = uni_dist(generator);
        for (size_t j = 0; j < data_dim; ++j) {
            queries[i * data_dim + j] = dataset[e * data_dim + j] + query_dist(generator);
        }
        gt[i] = e;
    }

    fvecs_write("data.vecs", data_dim, dataset_size, dataset.data());
    fvecs_write("query.vecs", data_dim, query_size, queries.data());
    ivecs_write("gt.vecs", 1, query_size, gt.data());
}
