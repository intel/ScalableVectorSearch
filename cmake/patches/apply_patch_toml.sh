TOML_PATCH=$1
# First check if the patch can be reversed (i.e. was it applied earlier)
# Apply it only when it's not applied earlier
if ! git apply -R --ignore-whitespace ${TOML_PATCH} --check; then git apply --ignore-whitespace ${TOML_PATCH}; fi
