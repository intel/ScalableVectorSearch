# SVS 0.0.4 Release Notes

## Major Changes

### Serialization Update

The serialization strategy for all SVS serialized objects has been updated from `v0.0.1` to
`v0.0.2`. In order to be compatible, all previously saved objects will need to be updated.

Updating can be done using the `pysvs` upgrade tool:
```python
import pysvs
pysvs.upgrader.upgrade("path-to-save-directory")
```
Some notes about the upgrade process are given below.

* If the object does not need upgrading, no changes will be made.
* If the object *does* need upgrading, then only the TOML file will be modified. The upgrade
  tool fill first create a backup (by changing the extension to `.backup.toml`) and
  then attempt an upgrade.

  If the upgrade fails, please let the maintainers know so we can fix the upgrade tool.

  Furthermore, if a backup file already exists, the upgrade process will abort.
* Objects upgraded to the `v0.0.2` serialization format are *forwards* compatible.
  This means that they can still be loaded by *older* version of `pysvs`.

**Why is this change needed?**

This change is necessary to support efficient introspective loading, where serialized
objects can be inspected for load compatibility. This, in turn, enables automatic loading
of previously serialized SVS objects.

## `pysvs` (Python)

### Additions and Changes

* Reloading a previously saved index no longer requires exact reconstruction of the original
  loader.

  Previously, if an index was constructed and saved using the following
  ```python
  # Load data using online compression.
  loader = pysvs.VectorDataLoader(...)
  lvq = pysvs.LVQLoader(loader, primary = 4, residual = 8)

  # Build the index.
  parameters = pysvs.VamanaBuildParameters(...)
  index = pysvs.Vamana.build(params, lvq, pysvs.L2, num_threads = 10)

  # Save the result to three directories.
  index.save("config", "graph", "data")
  ```
  Then the index must be reloaded using
  ```python
  lvq = pysvs.LVQLoader("data", primary = 4, residual = 8)
  index = pysvs.Vamana("config", pysvs.GraphLoader("graph"), lvq, distance = pysvs.L2)
  ```
  Now, the following will work
  ```python
  index = pysvs.Vamana(
      "config",
      "graph",   # No longer need explicit `pysvs.GraphLoader`
      "data",    # SVS discovers this is LVQ data automatically
      distance = pysvs.L2
  )
  ```
  To tailor the run-time parameters of reloaded data (for example, the strategy and padding
  used by LVQ), automatic inference of identifying parameters makes this easier:
  ```python
  lvq = pysvs.LVQLoader(
      "data",   # Parameters `primary`, `residual`, and `dims` are discovered automatically
      strategy = pysvs.LVQStrategy.Sequential,
      padding = 0
  )
  index = pysvs.Vamana("config", "graph", loader)
  ```

* Added an upgrade tool `pysvs.upgrader.upgrade` to upgrade the serialization layout of SVS
  objects.

## `libsvs` (C++)

### Changes

* Overhauled object loading. Context free classes should now accept a
  `svs::lib::ContextFreeLoadTable` and contextual classes should take a
  `svs::lib::LoadTable`.

  While most of the top level API remains unchanged, users are encouraged to look at the
  at the definitions of these classes in `include/svs/lib/saveload/load.h` to understand
  their capabilities and API.
* Added a new optional loading function `try_load -> svs::lib::Expected` which tries to load
  an object from a table and fails gracefully without an exception if it cannot.

  This API enables discovery and matching of previously serialized object, allowing
  implementation of the auto-loading functionality in `pysvs`.
