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

"""Dependency detection, paired graph-metrics runs, and PR reporting."""

import argparse
import copy
import html
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import urllib.request
import uuid


TARGET = "graph_metrics"
TOOL_DIRECTORY = Path(__file__).resolve().parent
COMMENT_MARKER = "<!-- svs-graph-metrics -->"


def command(args, **kwargs):
    return subprocess.check_output([str(arg) for arg in args], text=True, **kwargs).strip()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def output(name, value):
    if "GITHUB_OUTPUT" in os.environ:
        delimiter = uuid.uuid4().hex
        with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
            stream.write(f"{name}<<{delimiter}\n{value}\n{delimiter}\n")


def cpu_count():
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1


def compiler():
    selected = os.environ["CXX"]
    executable = shutil.which(selected)
    if executable is None:
        raise RuntimeError(f"Configured compiler is unavailable: {selected}")
    return executable


def configuration(head):
    path = (head / os.environ["GRAPH_METRICS_CONFIG"]).resolve()
    config = json.loads(path.read_text())
    for dataset in config["datasets"]:
        if dataset["source"]["type"] != "synthetic":
            raise ValueError("This workflow requires synthetic datasets")
        dataset["source"]["path"] = str((path.parent / dataset["source"]["path"]).resolve())
    return config


def logged(args, log, cwd=None):
    with log.open("a") as stream:
        command_line = "$ " + shlex.join([str(arg) for arg in args])
        print(command_line, file=stream, flush=True)
        print(command_line, flush=True)
        with subprocess.Popen(
            args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace", bufsize=1,
        ) as process:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)
            returncode = process.wait()
        if returncode:
            print(f"Command failed; diagnostic log: {log}", file=sys.stderr, flush=True)
            raise subprocess.CalledProcessError(returncode, args)


def configure(source, build, log, extra_args):
    query = build / ".cmake/api/v1/query"
    query.mkdir(parents=True, exist_ok=True)
    for name in ("codemodel-v2", "cmakeFiles-v1"):
        (query / name).touch()
    logged([
        os.environ.get("CMAKE", "cmake"),
        "-S", str(TOOL_DIRECTORY), "-B", str(build), "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_CXX_COMPILER={compiler()}",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        f"-DSVS_SOURCE_DIR={source}",
        *extra_args,
    ], log)


def cmake_reply(build, kind):
    reply = build / ".cmake/api/v1/reply"
    index = json.loads(sorted(reply.glob("index-*.json"))[-1].read_text())
    return json.loads((reply / index["reply"][kind]["jsonFile"]).read_text())


def project_dependencies(source, head, build, log):
    """Read CMake inputs and preprocess every TU in the calculator target closure."""
    model = cmake_reply(build, "codemodel-v2")
    source_root = Path(model["paths"]["source"])
    targets = {item["id"]: item for item in model["configurations"][0]["targets"]}
    pending = [item["id"] for item in targets.values() if item["name"] == TARGET]
    if not pending:
        raise RuntimeError(f"CMake did not define {TARGET}")
    visited, translation_units = set(), set()
    while pending:
        target_id = pending.pop()
        if target_id in visited:
            continue
        visited.add(target_id)
        target = json.loads((build / ".cmake/api/v1/reply" / targets[target_id]["jsonFile"]).read_text())
        pending.extend(item["id"] for item in target.get("dependencies", []))
        for item in target.get("sources", []):
            if "compileGroupIndex" in item:
                origin = build if item.get("isGenerated") else source_root
                translation_units.add((origin / item["path"]).resolve())

    inputs = cmake_reply(build, "cmakeFiles-v1")
    files = {(Path(inputs["paths"]["source"]) / item["path"]).resolve() for item in inputs["inputs"]}
    commands = json.loads((build / "compile_commands.json").read_text())
    scanned = set()
    for entry in commands:
        unit = (Path(entry["directory"]) / entry["file"]).resolve()
        if unit not in translation_units:
            continue
        arguments = entry.get("arguments") or shlex.split(entry["command"])
        filtered, skip = [], False
        for argument in arguments:
            if skip:
                skip = False
            elif argument in ("-o", "-MF", "-MT", "-MQ"):
                skip = True
            elif argument not in ("-c", "-MD", "-MMD"):
                filtered.append(argument)
        depfile = build / f"graph-metrics-{len(scanned)}.d"
        # -M only preprocesses; detection does not compile or run the benchmark.
        logged(filtered + ["-M", "-MF", str(depfile), "-MT", "graph_metrics"],
               log, cwd=entry["directory"])
        dependencies = depfile.read_text().replace("\\\n", " ").partition(":")[2]
        files.update(
            (Path(entry["directory"]) / name.replace("$$", "$")).resolve()
            for name in shlex.split(dependencies)
        )
        scanned.add(unit)
    if scanned != translation_units:
        raise RuntimeError(f"Missing compile commands for {sorted(map(str, translation_units - scanned))}")

    relative = set()
    for root in (source, head):
        for path in files:
            if path.is_relative_to(root):
                relative.add(path.relative_to(root).as_posix())
    return relative


def workflow_dependencies(root, tool_relative):
    # Discover the workflow support directory and the workflows referring to it,
    # rather than maintaining a list of C++ or workflow dependency filenames.
    result = set()
    for path in (root / tool_relative).rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            result.add(path.relative_to(root).as_posix())
    for path in (root / ".github/workflows").glob("*"):
        if path.suffix in (".yml", ".yaml") and tool_relative.as_posix() + "/" in path.read_text():
            result.add(path.relative_to(root).as_posix())
    return result


def detect(args):
    args.result.mkdir(parents=True, exist_ok=True)
    base_sha = command(["git", "-C", args.base, "rev-parse", "HEAD"])
    head_sha = command(["git", "-C", args.head, "rev-parse", "HEAD"])
    record = {"base_sha": base_sha, "head_sha": head_sha, "status": "failed"}
    try:
        # Three-dot diff matches the files changed by the PR, including deletions.
        changed = set(command([
            "git", "-C", args.head, "diff", "--name-only", "--no-renames", "-z",
            f"{base_sha}...{head_sha}",
        ]).split("\0")) - {""}
        dependencies = set()
        tool_relative = TOOL_DIRECTORY.relative_to(args.head)
        for label, source in (("base", args.base), ("head", args.head)):
            build = args.work / label
            log = args.result / f"{label}-dependencies.log"
            configure(source, build, log, args.cmake_arg)
            dependencies.update(project_dependencies(source, args.head, build, log))
            dependencies.update(workflow_dependencies(source, tool_relative))
        matched = sorted(changed & dependencies)
        needed = bool(matched) or os.environ.get("FORCE_RUN") == "true"
        record.update(
            status="needed" if needed else "skipped",
            changed_files=sorted(changed),
            dependencies=sorted(dependencies),
            matched_files=matched,
            compiler=command([compiler(), "--version"]).splitlines()[0],
        )
        output("needed", str(needed).lower())
        output("base_sha", base_sha)
        output("head_sha", head_sha)
        message = ("Graph metrics required: " + (", ".join(matched) or "manual run")
                   if needed else "Graph metrics skipped: no changed files affect the calculator or workflow.")
        print(message)
        if "GITHUB_STEP_SUMMARY" in os.environ:
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as stream:
                stream.write(message + "\n")
    finally:
        write_json(args.result / "detection.json", record)


def paths(args):
    cache_paths = []
    for dataset in configuration(args.head)["datasets"]:
        path = dataset["source"]["path"]
        cache_paths.extend([path, path + ".meta.json"])
    output("cache_paths", "\n".join(cache_paths))


def calculate(args):
    args.result.mkdir(parents=True, exist_ok=True)
    config = configuration(args.head)
    record = {
        "base_sha": command(["git", "-C", args.base, "rev-parse", "HEAD"]),
        "head_sha": command(["git", "-C", args.head, "rev-parse", "HEAD"]),
        "compiler": command([compiler(), "--version"]).splitlines()[0],
        "status": "failed",
    }
    try:
        # Both revisions run sequentially on the same machine, with the same
        # calculator, configuration and persisted vectors.
        for label, source in (("base", args.base), ("head", args.head)):
            build = args.work / label
            print(f"{label}: configuring revision {record[f'{label}_sha']}", flush=True)
            configure(source, build, args.result / f"{label}-build.log", args.cmake_arg)
            print(f"{label}: compiling {TARGET}", flush=True)
            logged([
                os.environ.get("CMAKE", "cmake"), "--build", str(build),
                "--target", TARGET, "--parallel", str(cpu_count()),
            ], args.result / f"{label}-build.log")
            effective = copy.deepcopy(config)
            effective["output_json"] = str(args.result / f"{label}.json")
            # Keep progress on stderr so logged() can stream it and save the artifact.
            effective.pop("output_log", None)
            config_path = args.result / f"{label}-config.json"
            write_json(config_path, effective)
            print(f"{label}: calculating graph metrics", flush=True)
            logged([str(build / TARGET), "--config", str(config_path)],
                   args.result / f"{label}-metrics.log")
        reports = [json.loads((args.result / f"{label}.json").read_text()) for label in ("base", "head")]
        identities = [
            {item["dataset_name"]: item["dataset_parameters"]["sha256"] for item in report["datasets"]}
            for report in reports
        ]
        if identities[0] != identities[1]:
            raise RuntimeError("Base and PR results used different vector data")
        record["status"] = "completed"
    finally:
        write_json(args.result / "run.json", record)


class GitHub:
    def __init__(self):
        self.root = os.environ.get("GITHUB_API_URL", "https://api.github.com")
        self.repository = os.environ["GITHUB_REPOSITORY"]

    def request(self, method, path, body=None):
        request = urllib.request.Request(
            self.root + path,
            data=json.dumps(body).encode() if body is not None else None,
            method=method,
            headers={
                "Authorization": "Bearer " + os.environ["GITHUB_TOKEN"],
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)

    def pages(self, path, key=None):
        result = []
        for page in range(1, 100):
            reply = self.request("GET", f"{path}?per_page=100&page={page}")
            items = reply[key] if key else reply
            result.extend(items)
            if len(items) < 100:
                return result
        raise RuntimeError("GitHub pagination limit exceeded")


def artifact_json(path):
    # Artifacts are untrusted input to the privileged comment job.
    if path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("Metrics artifact is too large")
    with path.open() as stream:
        return json.load(stream)


def artifact_record(path):
    if not path.is_file():
        return None
    try:
        record = artifact_json(path)
        if not isinstance(record, dict) or not isinstance(record.get("status"), str):
            raise ValueError("Invalid status record")
        for key in ("head_sha", "base_sha"):
            if not isinstance(record.get(key), str) or not re.fullmatch(r"[0-9a-f]{40}", record[key]):
                raise ValueError("Invalid commit identity")
        return record
    except (ValueError, OSError):
        print(f"Ignoring invalid metadata artifact: {path.name}")
        return None


def text(value):
    value = re.sub(r"[\x00-\x1f\x7f]", " ", str(value)[:200])
    value = html.escape(value, quote=True).replace("|", "&#124;").replace("@", "&#64;")
    return re.sub(r"([\\`*_\[\]()!])", r"\\\1", value)


def number(value, digits=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("Invalid numeric metric")
    if digits is None:
        if not isinstance(value, int):
            raise ValueError("Expected an integer metric")
        return str(value)
    return f"{value:.{digits}f}"


def cells(build):
    vertex = build["max_incoming_asymmetry_count_vertex"]
    incoming = number(build["max_incoming_asymmetry_count"])
    if vertex is not None:
        incoming += " @ vector " + number(vertex["vector_index"])
    return [
        number(build["unreachable_vertices"]),
        number(build["edge_asymmetry_fraction"], 6),
        "[" + ", ".join(number(value) for value in build["entry_points_internal"]) + "]",
        incoming,
        number(build["average_shortest_path_hops"], 3),
        number(build["maximum_shortest_path_hops"]),
        number(build["index_build_time_seconds"], 3),
    ]


def comparison(base, head):
    identities = [
        {
            dataset["dataset_name"]: (
                dataset["dataset_parameters"]["sha256"],
                dataset["dataset_parameters"]["selected_vectors"],
                dataset["dataset_parameters"]["dimensions"],
                dataset["dataset_parameters"]["distance"],
            )
            for dataset in report["datasets"]
        }
        for report in (base, head)
    ]
    if identities[0] != identities[1]:
        raise ValueError("Base and PR dataset identities differ")
    before = {
        (dataset["dataset_name"], index["index_type"], build["build_method"]): build
        for dataset in base["datasets"]
        for index in dataset["indexes"]
        for build in index["builds"]
    }
    lines = ["Values show **base → PR**. Incoming-asymmetry vertices use source vector indices."]
    for dataset in head["datasets"]:
        for index in dataset["indexes"]:
            lines.extend([
                "", f"**{text(dataset['dataset_name'])} — {text(index['index_type'])}**", "",
                "| Build method | Unreachable | Asymmetry fraction | Entry points | Max incoming asymmetry (vertex) | Avg shortest path | Max shortest path | Build time (s) |",
                "|---|---:|---:|---|---|---:|---:|---:|",
            ])
            for build in index["builds"]:
                key = (dataset["dataset_name"], index["index_type"], build["build_method"])
                previous = cells(before[key])
                current = cells(build)
                values = [text(build["build_method"])] + [
                    f"{old} → {new}" for old, new in zip(previous, current)
                ]
                lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def comment(args):
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    run = event.get("workflow_run")
    if (run is not None and run["event"] != "pull_request") or (
        run is None and "pull_request" not in event
    ):
        print("No PR comment: this event is not a pull request.", flush=True)
        return
    api = GitHub()
    prefix = f"/repos/{api.repository}"
    if run is None:
        pr = event["pull_request"]
        if pr["head"]["repo"]["full_name"] != api.repository:
            print("Fork PR comments are handled by the default-branch reporter.", flush=True)
            return
        run = api.request("GET", f"{prefix}/actions/runs/{int(os.environ['GITHUB_RUN_ID'])}")
        # The current workflow has not finished, but both producer jobs have.
        run["conclusion"] = os.environ["GRAPH_METRICS_CONCLUSION"]
        run["pull_requests"] = [pr]
    # Resolve the PR through GitHub's event/run association, never an artifact's
    # claimed PR number. Only current PR heads may update the comment.
    run_prs = run.get("pull_requests") or []
    candidates = run_prs or api.pages(f"{prefix}/commits/{run['head_sha']}/pulls")
    if not candidates:
        print(f"No PR associated with graph-metrics run {run['id']}.", flush=True)
        return
    run_heads = {item["number"]: item.get("head", {}).get("sha", run["head_sha"]) for item in run_prs}
    artifacts = api.pages(f"{prefix}/actions/runs/{run['id']}/artifacts", "artifacts")
    detection_path = args.artifacts / "graph-metrics-results-detection/detection.json"
    result_dir = args.artifacts / "graph-metrics-results-run"
    detection = artifact_record(detection_path)
    result_path = result_dir / "run.json"
    result = artifact_record(result_path)
    for candidate in candidates:
        pr = api.request("GET", f"{prefix}/pulls/{int(candidate['number'])}")
        associated_sha = run_heads.get(candidate["number"], run["head_sha"])
        if pr["state"] != "open" or pr["head"]["sha"] != associated_sha:
            print(f"No comment on PR #{pr['number']}: closed or its head has changed.", flush=True)
            continue
        current_sha = pr["head"]["sha"]
        if any(record and record.get("head_sha") != current_sha for record in (detection, result)):
            print(f"No comment on PR #{pr['number']}: artifact revision does not match.", flush=True)
            continue
        base_sha = (detection or result or {}).get("base_sha", "")
        if not re.fullmatch(r"[0-9a-f]{40}", base_sha):
            base_sha = pr["base"]["sha"]
        run_url = run["html_url"]
        lines = [
            COMMENT_MARKER,
            f"<!-- run={int(run['id'])} attempt={int(run.get('run_attempt', 1))} -->",
            "### Synthetic graph metrics", "",
            f"Base `{base_sha[:12]}` → PR `{current_sha[:12]}`.",
        ]
        if detection and detection["status"] == "skipped" and run["conclusion"] == "success":
            lines.extend(["", "Skipped: no changed files affect the calculator or workflow dependencies."])
        elif result and result["status"] == "completed" and run["conclusion"] == "success":
            try:
                summary = comparison(
                    artifact_json(result_dir / "base.json"),
                    artifact_json(result_dir / "head.json"),
                )
                lines.extend(["", f"Compiler: {text(result['compiler'])}.", "", summary])
            except (KeyError, TypeError, ValueError, OSError):
                lines.extend(["", "The metrics report could not be validated. See the run and artifacts."])
        else:
            lines.extend(["", f"Graph metrics are unavailable (run: {text(run['conclusion'])}). See the run logs."])
        links = ["", f"[Workflow run and logs]({run_url})"]
        for artifact in artifacts:
            if artifact["name"] == "graph-metrics-results-run" and not artifact.get("expired"):
                artifact_url = f"{run_url}/artifacts/{int(artifact['id'])}"
                links.append(f"[Full base/PR JSON reports, configurations, and logs]({artifact_url})")
        body = "\n".join(lines)[:60000] + "\n" + "\n".join(links)
        existing = next((
            item for item in api.pages(f"{prefix}/issues/{pr['number']}/comments")
            if item["user"]["login"] == "github-actions[bot]" and COMMENT_MARKER in item["body"]
        ), None)
        if existing:
            sequence = re.search(r"<!-- run=(\d+) attempt=(\d+) -->", existing["body"])
            current = (int(run["id"]), int(run.get("run_attempt", 1)))
            if sequence and tuple(map(int, sequence.groups())) > current:
                print(f"PR #{pr['number']} already has results from a newer run.", flush=True)
                continue
            if existing["body"] == body:
                print(f"PR #{pr['number']} already has this graph-metrics comment.", flush=True)
                continue
            api.request("PATCH", f"{prefix}/issues/comments/{existing['id']}", {"body": body})
            print(f"Updated graph-metrics comment on PR #{pr['number']}.", flush=True)
        else:
            api.request("POST", f"{prefix}/issues/{pr['number']}/comments", {"body": body})
            print(f"Created graph-metrics comment on PR #{pr['number']}.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("detect", "calculate"):
        sub = commands.add_parser(name)
        sub.add_argument("--head", type=Path, required=True)
        sub.add_argument("--base", type=Path, required=True)
        sub.add_argument("--work", type=Path, required=True)
        sub.add_argument("--result", type=Path, required=True)
        sub.add_argument("--cmake-arg", action="append", default=[])
    sub = commands.add_parser("paths")
    sub.add_argument("--head", type=Path, required=True)
    sub = commands.add_parser("comment")
    sub.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    for key in ("head", "base", "work", "result", "artifacts"):
        if hasattr(args, key):
            setattr(args, key, getattr(args, key).resolve())
    {"detect": detect, "calculate": calculate, "paths": paths, "comment": comment}[args.action](args)


if __name__ == "__main__":
    main()
