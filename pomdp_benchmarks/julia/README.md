# Julia SARSOP Bridge

This folder contains `sarsop_bridge.jl`, used by the Python benchmark suite to
optionally solve tabular POMDPs with `SARSOP.jl` and import the resulting
alpha-vector policy back into Python.

## Required Julia packages

```julia
using Pkg
Pkg.add(["POMDPs", "POMDPTools", "SARSOP", "JSON3"])
```

## How Python calls it

The benchmark runner invokes:

```bash
julia pomdp_benchmarks/julia/sarsop_bridge.jl \
  --model-json <tmp-model.json> \
  --output-json <tmp-policy.json> \
  --timeout-sec 120 \
  --precision 1e-3
```

If Julia or these packages are missing, the `SARSOPJulia` solver will be
reported as `skipped` in benchmark output rather than crashing the run.
