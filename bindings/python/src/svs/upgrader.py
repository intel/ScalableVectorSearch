# Copyright 2024 Intel Corporation
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

# A utility for upgrading serialized SVS objects.
import toml
import os
import shutil
from pathlib import Path

from .loader import library
_lib = library()

DEFAULT_SCHEMA_FILE = "serialization.toml"

def _reformat_toml(path):
    _lib.__reformat_toml(path)

def _is_reserved(key: str):
    return key.startswith("__")

def _resolve_schema(arg):
    if arg is None:
        return toml.load(Path(__file__).parent / DEFAULT_SCHEMA_FILE)

    if isinstance(arg, os.PathLike):
        return  toml.load(arg)

    return arg


def _parse_type(typ: str):
    if typ == "str":
        return str
    if typ == "int":
        return int
    if typ == "float":
        return float
    if typ == "bool":
        return bool
    if typ == "table":
        return dict
    if typ == "array":
        return list

    raise Exception(f"Encountered unknown type: {typ}!")

def _parse_type_list(types):
    if isinstance(types, str):
        return {_parse_type(types)}
    if isinstance(types, list):
        return {_parse_type(t) for t in types}

    raise Exception(f"Unhandled types collection: {types}")

class LeafType:
    def __init__(self, source):
        # If the source is just a string, then assume it only contains a single valid type.
        if isinstance(source, str) or isinstance(source, list):
            self.types = _parse_type_list(source)
            self.optional = False
            return

        # If the source is a list, assume it is a list of valid types.
        if isinstance(source, dict):
            self.types = _parse_type_list(source["type"])
            self.optional = source.get("optional", False)
            return

        raise Exception(f"Cannot construct a leaf type from {source}!")

    def is_match(self, target):
        """
        Return ``True`` if the object or type ``target`` is one of the types accepted by
        this leaf element. Otherwise, return ``False``.
        """
        if isinstance(target, type):
            return target in self.types

        return type(target) in self.types


class Schema:
    """
    A class used to match a named Schema prototype with an untyped dictionary that contains
    a serialized SVS object.

    Schema are injested as dictionaries.
    """
    def __init__(self, source: dict):
        # Paranoid type check
        assert(isinstance(source, dict))

        self.name = source["__schema__"]
        self.version = source["__version__"]
        self.leaves = {k : LeafType(v) for (k, v) in source.items() if not _is_reserved(k)}

    def is_match(self, target):
        """
        Prerequisites:
        * ``target`` is a dictionary containing at least a ``__version__`` key.
        """
        # Paranoid type check
        assert(isinstance(target, dict))

        # Prerequisite: `__version__` key exists
        if target["__version__"] != self.version:
            return False

        target_keys = {k for k in target.keys() if not _is_reserved(k)}

        # Check if the keys of the target are a subset of all the keys associated with this
        # schema.
        if not target_keys.issubset(self.all_keys()):
            return False

        # Next - check that the target contains all keys required by the schema.
        if not self.required_keys().issubset(target_keys):
            return False

        # Key check has succeeded.
        # Finally, verify that the types of the elements in the dictionary match the
        # expected types for this schema.
        for k in target_keys:
            if not self.leaves[k].is_match(target[k]):
                return False

        # All checks passed - it's a match.
        return True

    def all_keys(self):
        """
        Return all possible keys that can be part of the schema.
        """
        return self.leaves.keys()

    def required_keys(self):
        """
        Return the required keys for this schema.
        """
        return {k for (k, v) in self.leaves.items() if not v.optional}

class SchemaList:
    def __init__(self, source):
        # If this is not a dictionary - assume it is a path and use it to create the source
        # dictionary.
        if not isinstance(source, dict):
            source = toml.load(source)

        self.version = source["schema_version"]
        self.schemas = [Schema(x) for x in source["schemas"]]

    def match(self, target):
        matches = [schema for schema in self.schemas if schema.is_match(target)]
        num_matches = len(matches)
        if num_matches == 0:
            raise Exception(f"Could not match target {target}!");
        if num_matches != 1:
            raise Exception(f"Found {num_matches} matches for {target}!");

        return matches[0]

    def transform(self, target):
        # Recursively transform each element in the list.
        if isinstance(target, list):
            for t in target:
                self.transform(t)

            return

        # If this is a dictionary, transform each non-reserved member before matching this
        # level.
        if isinstance(target, dict):
            assert("__schema__" not in target)
            for t in target.values():
                self.transform(t)
            schema = self.match(target)
            target["__schema__"] = schema.name
            return

        # For all other types - we hit the end of recursion.

#####
##### main upgrade function
#####

def _upgrade_v0_0_1(path, schema, obj, backup = True, _assume_version = None):
    # Try to upgrade the object.
    matcher = SchemaList(_resolve_schema(schema))

    # If we are specified to not assume the version - then assume this has a regular object
    # layout.
    if _assume_version is None:
        try:
            assert(obj["__version__"] == "v0.0.1")
            matcher.transform(obj["object"])
            obj["__version__"] = "v0.0.2"
        except:
            print(f"Error upgrading file {path} to version v0.0.2!")
            raise

    elif _assume_version == "v0.0.1":
        try:
            for v in obj.values():
                matcher.transform(v)
        except:
            print(f"Error upgrading file {path} to version v0.0.2!")
            raise
    else:
        raise Exception(
            f"Unhandled version {_assume_version} in compatibility path!"
            f"Please report to the maintainers."
        )

    # Matching was successful!
    # Create a backup (if requested) and save the new object.
    if backup:
        prefix, ext = os.path.splitext(path)
        backuppath = prefix + ".backup" + ext
        # Error if a backup exists.
        if os.path.exists(backuppath):
            raise Exception(f"Backup file {backuppath} already exists. Aborting!")

        print(f"Creating a backup of {path} to {backuppath}")
        shutil.copy(path, backuppath)

    # Using Python's TOML serializer leaves the resulting file in an ugly state.
    # Use the C++ TOML serializer on the final file to put it back into a nicer format.
    with open(path, "w") as f:
        toml.dump(obj, f)

    _reformat_toml(path)

def upgrade(
    path,
    schema_path = None,
    backup = True,
    # Assume the version of the source object rather than deducing it.
    _assume_version = None,
):
    """
    Upgrade a serialized SVS object to the most recent global serialization version.
    Currently, this tool only modifies SVS TOML files.
    Furthermore, this tool only upgrades the global serialization strategy (how objects
    in general are stored). It does not (yet) support upgrading individual objects whose
    serialization version has changed.

    Objects that are already at the current serialization version are left unchanged.

    Supported Upgrade Paths:
        * ``v0.0.1`` to ``v0.0.2``.

    Args:
        path: The path to the serialized SVS object.

            This will usually be directory containing a canonically serialized SVS object
            (identifiable by the existence of a file called ``svs_config.toml`` and
            potentially other auxiliary files.

            Alternatively, this can be a path to an SVS serialized file directly.

    Keyword Args:
        backup: Create a backup of the original object before creating changes.
            A backed up TOML file will have its extension modified to ``.backup.toml``.
            To restore a backup, remove the ``.backup`` infix.

        schema_path: Path to a schema file describing the schemas of SVS objects.
            This is used during the upgrade from global serialization version ``v0.0.1`` to
            ``v0.0.2`` and will be supplied by default.

        _assume_version: Non-conforming SVS objects may not support automatic global
            version detection. The assumed version (as a string triple like ``"v0.0.1"``)
            may be provided.
    """
    # If we are given a directory - try to resolve it as a path to a TOML file.
    if os.path.isdir(path):
        path = os.path.join(path, "svs_config.toml")

    # Parse the source TOML file.
    obj = toml.load(path)
    version = obj.get("__version__", _assume_version)
    if version is None:
        raise Exception(f"Cannot determine the version of SVS file {path}")

    if version == "v0.0.1":
        return _upgrade_v0_0_1(
            path, schema_path, obj, backup = backup, _assume_version = _assume_version
        )

    print(f"File {path} is up to date!")

