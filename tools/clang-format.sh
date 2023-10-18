#!/bin/sh

# Allow users to supply a custom path to `clang-format`
CLANGFORMAT="${1:-clang-format}"
DIRECTORIES=( "bindings/python/src" "include" "benchmark" "tests" "utils" "examples/cpp" )

for i in "${DIRECTORIES[@]}"
do
    find "./$i" \( -iname "*.h" -o -iname "*.cpp" \) ! -iname "*toml_impl.h" | xargs "$CLANGFORMAT" -i
done

