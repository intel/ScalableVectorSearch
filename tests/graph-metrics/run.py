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

"""Run the C++ graph calculator with JSON configuration using Python's standard library."""

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile


MAX_SIZE = (1 << (struct.calcsize("P") * 8)) - 1
# The existing C++ TOML parser uses signed 64-bit integers for prepared settings.
MAX_INTEGER = min(MAX_SIZE, (1 << 63) - 1)
MAX_FLOAT32 = float.fromhex("0x1.fffffep+127")
ELEMENT_SIZES = {"uint8": 1, "int8": 1, "float16": 2, "float32": 4}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    def object_pairs(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, f"Duplicate JSON key: {key}")
            value[key] = item
        return value

    def invalid_constant(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    with path.open() as stream:
        return json.load(stream, object_pairs_hook=object_pairs, parse_constant=invalid_constant)


def object_keys(value, context, required, optional=()):
    require(isinstance(value, dict), f"{context} must be an object")
    for key in required:
        require(key in value, f"Missing {context}.{key}")
    for key in value:
        require(key in required or key in optional, f"Unknown field {context}.{key}")


def text(value, context):
    require(isinstance(value, str) and value and "\0" not in value,
            f"{context} must be a nonempty string without NUL")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{context} must contain valid Unicode") from error
    return value


def integer(value, context, minimum=1, maximum=MAX_INTEGER):
    require(type(value) is int and minimum <= value <= maximum,
            f"{context} must be an integer in [{minimum}, {maximum}]")
    return value


def number(value, context):
    require(type(value) in (int, float) and -MAX_FLOAT32 <= value <= MAX_FLOAT32,
            f"{context} must be a finite number in [-{MAX_FLOAT32}, {MAX_FLOAT32}]")
    return float(value)


def boolean(value, context):
    require(type(value) is bool, f"{context} must be a boolean")
    return value


def resolve(directory, value):
    return (directory / text(value, "path")).resolve()


def same_path(left, right):
    return left == right or (left.exists() and right.exists() and left.samefile(right))


def sidecars(path):
    return Path(str(path) + ".meta.json"), Path(str(path) + ".lock")


def read_build_methods(methods, graph_type):
    require(isinstance(methods, list) and methods, "build_methods must be a nonempty array")
    seen = set()
    for method in methods:
        name = text(method, "build_methods entry")
        require(name in ("sample_by_sample", "single_batch", "batches"),
                f"Unsupported build method {name!r}; expected sample_by_sample, single_batch, or batches")
        require(name not in seen, f"Duplicate build method: {name}")
        seen.add(name)
    require(graph_type != "non_concurrent" or seen != {"sample_by_sample"},
            "sample_by_sample requires a concurrent index")
    return methods


def read_build_parameters(source):
    threads = source["threads"]
    if threads == "auto":
        threads = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    threads = integer(threads, "threads")
    graph_type = source["graph_type"]
    require(graph_type in ("concurrent", "non_concurrent", "both"),
            f"graph_type must be concurrent, non_concurrent, or both; got {graph_type!r}")
    methods = read_build_methods(source["build_methods"], graph_type)
    build = source["build_parameters"]
    object_keys(build, "build_parameters", ("degree", "window", "max_candidates", "alpha"),
                ("use_full_search_history",))
    degree = integer(build["degree"], "degree", maximum=(1 << 32) - 2)
    window = integer(build["window"], "window")
    candidates = integer(build["max_candidates"], "max_candidates")
    require(candidates >= window >= degree, "Require max_candidates >= window >= degree")
    alpha = build["alpha"]
    object_keys(alpha, "alpha", ("L2", "MIP"))
    alphas = {key: number(value, f"alpha.{key}") for key, value in alpha.items()}
    require(alphas["L2"] >= 1 and 0 < alphas["MIP"] <= 1 and
            struct.unpack("<f", struct.pack("<f", alphas["MIP"]))[0] > 0,
            "Require alpha.L2 >= 1 and alpha.MIP in (0, 1], representable as float32")
    common = {
        "threads": threads,
        "graph_type": graph_type,
        "build_methods": methods,
        "batch_size": integer(source["batch_size"], "batch_size"),
        "recompute_entry_point": boolean(source["recompute_entry_point"], "recompute_entry_point"),
        "degree": degree,
        "window": window,
        "max_candidates": candidates,
        "use_full_search_history": boolean(build.get("use_full_search_history", True), "use_full_search_history"),
    }
    return common, alphas


def read_generation(data, total, dimensions):
    require("distribution" in data, "Missing source.distribution")
    distribution = data["distribution"]
    if distribution == "normal":
        fields = ("mean", "stddev")
    elif distribution == "uniform":
        fields = ("lower", "upper")
    else:
        raise ValueError(f"Unsupported distribution {distribution!r}; expected normal or uniform")
    object_keys(data, "source", ("type", "path", "distribution", "seed", *fields))
    parameters = {key: number(data[key], key) for key in fields}
    if distribution == "normal":
        require(parameters["stddev"] > 0, "stddev must be > 0")
    else:
        require(parameters["lower"] < parameters["upper"], "Require lower < upper")
    return {
        "schema_version": 1, "generator": "svs-mt19937-box-muller-v1",
        "vectors": total, "dimensions": dimensions, "data_type": "float32",
        "distribution": distribution, "parameters": parameters,
        "seed": integer(data["seed"], "seed", 0, (1 << 32) - 1),
    }


def read_dataset(dataset, directory, alphas, names):
    object_keys(dataset, "dataset", ("name", "vectors", "dimensions", "distance", "source"))
    name = text(dataset["name"], "dataset.name")
    require(name not in names, f"Duplicate dataset name: {name}")
    names.add(name)
    total = integer(dataset["vectors"], "vectors", maximum=(1 << 32) - 2)
    dimensions = integer(dataset["dimensions"], "dimensions", maximum=(1 << 32) - 1)
    require(total * (4 + 4 * dimensions) <= MAX_SIZE, "Vector data size overflows size_t")
    distance = dataset["distance"]
    require(distance in ("L2", "MIP"), f"distance must be L2 or MIP; got {distance!r}")
    data = dataset["source"]
    require(isinstance(data, dict) and "type" in data, "Missing source.type")
    kind = data["type"]
    generation = None
    if kind == "file":
        object_keys(data, "source", ("type", "path", "data_type"))
        data_type = text(data["data_type"], "source.data_type")
        require(data_type in ELEMENT_SIZES,
                f"Unsupported source type {data_type!r}; expected one of {tuple(ELEMENT_SIZES)}")
    elif kind == "synthetic":
        generation = read_generation(data, total, dimensions)
        data_type = "float32"
    else:
        raise ValueError(f"Unsupported source.type {kind!r}; expected file or synthetic")
    return {
        "name": name, "vectors": total, "dimensions": dimensions, "distance": distance,
        "alpha": alphas[distance], "path": resolve(directory, data["path"]), "kind": kind,
        "source_type": data_type, "generation": generation,
        "source_vectors": total, "sha256": "",
    }


def read_inputs(datasets, path, alphas):
    require(isinstance(datasets, list) and datasets, "datasets must be a nonempty array")
    names, generators, inputs = set(), {}, []
    protected = {path}
    for dataset in datasets:
        item = read_dataset(dataset, path.parent, alphas, names)
        data_path, generation = item["path"], item["generation"]
        protected.add(data_path)
        if generation is not None:
            require(data_path not in generators or generators[data_path] == generation,
                    f"Conflicting synthetic configurations for {data_path}")
            generators[data_path] = generation
        inputs.append(item)
    for data_path in generators:
        for sidecar in sidecars(data_path):
            require(not any(same_path(sidecar, item) for item in protected),
                    f"Synthetic sidecar collides with a dataset or configuration: {sidecar}")
    for data_path in generators:
        protected.update(sidecars(data_path))
    return inputs, protected


def read_output_paths(source, path, protected):
    output_json = resolve(path.parent, source["output_json"])
    output_log = resolve(path.parent, source["output_log"]) if "output_log" in source else None
    for output in (output_json, output_log):
        if output is not None:
            require(not any(same_path(output, item) for item in protected),
                    f"Output path aliases an input: {output}")
    require(output_log is None or not same_path(output_json, output_log),
            "JSON and log paths must differ and must not alias the same file")
    return output_json, output_log


def read_configuration(filename):
    path = filename.resolve(strict=True)
    source = read_json(path)
    object_keys(source, "config", (
        "threads", "graph_type", "build_methods", "batch_size", "recompute_entry_point",
        "build_parameters", "datasets", "output_json",
    ), ("output_log",))
    common, alphas = read_build_parameters(source)
    inputs, protected = read_inputs(source["datasets"], path, alphas)
    output_json, output_log = read_output_paths(source, path, protected)
    return {"path": path, "output_json": output_json, "output_log": output_log,
            "common": common, "inputs": inputs}


def toml_value(value):
    if isinstance(value, (str, Path)):
        # TOML basic strings share JSON escapes, but reject raw DEL and surrogate escapes.
        return json.dumps(str(value), ensure_ascii=False).replace("\x7f", "\\u007f")
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is int:
        return str(value)
    if type(value) is float and math.isfinite(value):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    raise ValueError(f"Unsupported prepared-setting value: {value!r}")


def write_prepared(path, config, output_json, inputs):
    values = {"protocol_version": 1, "configuration_file": config["path"],
              "output_json": output_json, **config["common"]}
    lines = [f"{key} = {toml_value(value)}" for key, value in values.items()]
    for item in inputs:
        generation = item["generation"]
        fields = {key: value for key, value in item.items() if key != "generation"}
        fields["generation_json"] = json.dumps(generation, sort_keys=True, allow_nan=False)
        if generation is not None:
            distribution = generation["distribution"]
            parameters = generation["parameters"]
            fields.update(seed=generation["seed"], distribution=distribution)
            fields["offset"] = parameters["mean"] if distribution == "normal" else parameters["lower"]
            fields["scale"] = (parameters["stddev"] if distribution == "normal"
                               else parameters["upper"] - parameters["lower"])
        lines.extend(["", "[[datasets]]"])
        lines.extend(f"{key} = {toml_value(value)}" for key, value in fields.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def selected_bytes(item):
    return item["vectors"] * (4 + ELEMENT_SIZES[item["source_type"]] * item["dimensions"])


def hash_prefix(path, length):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while length:
            block = stream.read(min(length, 1024 * 1024))
            require(block, f"Cannot read dataset for checksum: {path}")
            digest.update(block)
            length -= len(block)
    return digest.hexdigest()


def inspect_file(item):
    path = item["path"]
    with path.open("rb") as stream:
        header = stream.read(4)
    require(len(header) == 4, f"Cannot read dataset header: {path}")
    row_bytes = 4 + ELEMENT_SIZES[item["source_type"]] * item["dimensions"]
    length = path.stat().st_size
    require(struct.unpack("<I", header)[0] == item["dimensions"] and length % row_bytes == 0,
            f"Invalid vecs file dimensions or length: {path}")
    item["source_vectors"] = length // row_bytes
    require(item["source_vectors"] >= item["vectors"], f"Dataset has fewer rows than requested: {path}")
    item["sha256"] = hash_prefix(path, selected_bytes(item))


@contextlib.contextmanager
def temporary_file(destination):
    fd, name = tempfile.mkstemp(prefix=destination.name + ".tmp.", dir=destination.parent)
    os.close(fd)
    path = Path(name)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def prepare_synthetic(item, config, binary, work):
    path = item["path"]
    metadata, _ = sidecars(path)
    if path.exists() or metadata.exists():
        try:
            require(path.is_file() and metadata.is_file(),
                    "Incomplete synthetic dataset: both data and metadata must exist")
            saved = read_json(metadata)
            require(isinstance(saved, dict) and "sha256" in saved, "Missing synthetic SHA-256")
            expected = saved.pop("sha256")
            require(isinstance(expected, str) and re.fullmatch("[0-9a-f]{64}", expected),
                    "Invalid synthetic SHA-256")
            require(saved == item["generation"], "Synthetic metadata mismatch")
            require(path.stat().st_size == selected_bytes(item), "Synthetic dataset size mismatch")
            item["sha256"] = hash_prefix(path, selected_bytes(item))
            require(item["sha256"] == expected, "Synthetic dataset checksum mismatch")
            print(f"Reusing synthetic dataset: {path}", file=sys.stderr, flush=True)
            return
        except (ValueError, OSError) as error:
            print(f"Regenerating synthetic dataset: {path} ({error})", file=sys.stderr, flush=True)
    else:
        print(f"Generating synthetic dataset: {path}", file=sys.stderr, flush=True)
    # Keep the previous cache intact until generation and hashing both succeed.
    with temporary_file(path) as vectors, temporary_file(metadata) as sidecar:
        prepared = work / "generate.toml"
        write_prepared(prepared, config, work / "unused.json", [{**item, "path": vectors}])
        subprocess.run([str(binary), "--generate", str(prepared)],
                       stdout=sys.stderr, stderr=sys.stderr, check=True)
        require(vectors.stat().st_size == selected_bytes(item), "Generated dataset size mismatch")
        item["sha256"] = hash_prefix(vectors, selected_bytes(item))
        saved = {**item["generation"], "sha256": item["sha256"]}
        sidecar.write_text(json.dumps(saved, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(vectors, path)
        os.replace(sidecar, metadata)


def run(config, binary):
    output = config["output_json"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        # Hold locks through calculation so another launch cannot replace a cache in use.
        # Sorting prevents deadlock when two configurations share multiple datasets.
        paths = sorted({item["path"] for item in config["inputs"] if item["kind"] == "synthetic"})
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context(sidecars(path)[1].open("a+b"))
            fcntl.flock(lock, fcntl.LOCK_EX)
        work = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix=".graph-metrics-", dir=output.parent)))
        for item in config["inputs"]:
            if item["kind"] == "file":
                inspect_file(item)
        for item in config["inputs"]:
            if item["kind"] == "synthetic":
                prepare_synthetic(item, config, binary, work)
        prepared = work / "prepared.toml"
        report = work / "report.json"
        write_prepared(prepared, config, report, config["inputs"])
        subprocess.run([str(binary), "--prepared-config", str(prepared)],
                       stdout=sys.stderr, stderr=sys.stderr, check=True)
        for item in config["inputs"]:
            require(hash_prefix(item["path"], selected_bytes(item)) == item["sha256"],
                    f"Dataset changed during calculation: {item['path']}")
        read_json(report)
        os.replace(report, output)
    print(f"JSON report: {output}", file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    with contextlib.ExitStack() as stack:
        try:
            config = read_configuration(args.config)
            print(f"JSON: {config['output_json']}", file=sys.stderr, flush=True)
            if config["output_log"] is not None:
                log = config["output_log"]
                print(f"Log:  {log}", file=sys.stderr, flush=True)
                log.parent.mkdir(parents=True, exist_ok=True)
                stream = stack.enter_context(log.open("w"))
                stack.enter_context(contextlib.redirect_stderr(stream))
            run(config, args.binary.resolve(strict=True))
        except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
            print(f"Error: {error}", file=sys.stderr, flush=True)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
