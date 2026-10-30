# Vendored POT network simplex solver

Source: https://github.com/PythonOT/POT, tag `0.9.5`, directory `ot/lp/`.

Fetched unmodified from `https://raw.githubusercontent.com/PythonOT/POT/0.9.5/ot/lp/<file>`.

SHA-256:

```
4264bfe6760fcdf9b4c591e0e1fac9b2d983ccf6874dffb4d9e314868b197c93  network_simplex_simple.h
c16d242d385b9ad5c5b720d5a9b8526cb141b753f93eded9035368d0b04e66ec  full_bipartitegraph.h
359e9b1d580c1eb7213bfd961d57ea70f6713d364534ae343f6d79b90626d753  core.h
```

## Files

- `network_simplex_simple.h` — the exact transport solver behind `svs::distance::QEmd`.
- `full_bipartitegraph.h` — local header `network_simplex_simple.h` includes.
- `core.h` — local header `full_bipartitegraph.h` includes.

(`network_simplex_simple.h`'s other includes are all standard library headers.)

## Licenses

POT itself is MIT-licensed: "Copyright (c) 2016-2023 POT contributors".

All three files are Nicolas Bonneel's 2013 adaptation of LEMON
(`full_graph.h` / `network_simplex.h`) into a lighter mass-transport solver.
Each carries LEMON's original notice:

```
Copyright (C) 2003-2010
Egervary Jeno Kombinatorikus Optimalizalasi Kutatocsoport
(Egervary Research Group on Combinatorial Optimization, EGRES).

Permission to use, modify and distribute this software is granted
provided that this copyright notice appears in all copies. For
precise terms see the accompanying LICENSE file.

This software is provided "AS IS" with no warranty of any kind,
express or implied, and with no claim as to its suitability for any
purpose.
```

This permission notice must stay in each vendored file; do not strip it when editing.
