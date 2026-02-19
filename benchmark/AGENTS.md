# AGENTS.md — benchmark/

Benchmark harness and performance-eval code.

- Keep runs reproducible (inputs/config in repo).
- Separate harness changes from algorithm changes when possible.
- If shared with tests, validate impacted test targets.

## Performance benchmarking constraints
- **Warmup and iterations:** Do not change warmup rounds or iteration counts without justification and approval. These values are tuned for CI stability.
- **Input datasets:** Use fixed seeds and document dataset provenance. Do not commit large binary datasets without approval.
- **Baseline comparisons:** When adding new benchmarks, provide baseline comparisons against existing algorithms or library defaults.
- **Reporting:** Report median, min, max, stddev — not just mean. Flag outliers and variance sources (e.g., NUMA effects, Turbo Boost).
