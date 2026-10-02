#!/usr/bin/env bash
# Copyright 2026 Intel Corporation
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

set -euo pipefail

graph_metric_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

# Dataset definitions, graph parameters, thread count, and report paths live in JSON.
graph_metric_config="${1:-${graph_metric_root}/graph_metrics_config.json}"
graph_metric_build_dir="${graph_metric_root}/build"
graph_metric_build_type="Release"
graph_metric_build_jobs=8
graph_metric_cmake="auto" # Reuse the build directory's CMake, otherwise use PATH.
graph_metric_build_log="${graph_metric_build_dir}/graph-metrics/build.log"

if (( $# > 1 )); then
    printf 'Usage: bash .github/graph-metrics/graph_metric.sh [configuration.json]\n' >&2
    exit 2
fi
if [[ ! -f "${graph_metric_config}" ]]; then
    printf 'Configuration not found: %s\n' "${graph_metric_config}" >&2
    exit 2
fi

# Switching CMake installations in an existing build can invalidate FetchContent
# stamps and trigger dependency re-downloads. Keep its configured executable.
if [[ "${graph_metric_cmake}" == "auto" ]]; then
    graph_metric_cmake="cmake"
    if [[ -f "${graph_metric_build_dir}/CMakeCache.txt" ]]; then
        while IFS= read -r graph_metric_cache_line; do
            if [[ "${graph_metric_cache_line}" == CMAKE_COMMAND:INTERNAL=* ]]; then
                graph_metric_cached_cmake="${graph_metric_cache_line#*=}"
                if [[ -x "${graph_metric_cached_cmake}" ]]; then
                    graph_metric_cmake="${graph_metric_cached_cmake}"
                fi
                break
            fi
        done < "${graph_metric_build_dir}/CMakeCache.txt"
    fi
fi

printf 'Configuration: %s\nBuild log: %s\n' "${graph_metric_config}" "${graph_metric_build_log}"

# The calculator handles runtime JSON/log destinations from the config itself.
mkdir -p -- "$(dirname -- "${graph_metric_build_log}")"
{
    "${graph_metric_cmake}" -S "${graph_metric_root}" -B "${graph_metric_build_dir}" \
        -DCMAKE_BUILD_TYPE="${graph_metric_build_type}" -DSVS_BUILD_BINARIES=ON
    "${graph_metric_cmake}" --build "${graph_metric_build_dir}" --target graph_metrics \
        --parallel "${graph_metric_build_jobs}"
} > "${graph_metric_build_log}" 2>&1

exec "${graph_metric_build_dir}/utils/graph_metrics" --config "${graph_metric_config}"
