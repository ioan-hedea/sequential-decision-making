#!/usr/bin/env julia

using JSON3
using POMDPs
using SARSOP

# SparseCat lives in POMDPTools in modern stacks; keep fallback for older envs.
const _POMDP_TOOLS_OK = try
    @eval using POMDPTools
    true
catch
    false
end

if !_POMDP_TOOLS_OK
    @eval using POMDPModelTools
end

const _SparseCat = if _POMDP_TOOLS_OK
    POMDPTools.SparseCat
else
    POMDPModelTools.SparseCat
end

struct TabularBenchPOMDP <: POMDP{Int, Int, Int}
    T::Array{Float64, 3}  # [A, S, S]
    O::Array{Float64, 3}  # [A, S, O]
    R::Array{Float64, 2}  # [S, A]
    gamma::Float64
    init::Vector{Float64}
    states_vec::Vector{Int}
    actions_vec::Vector{Int}
    obs_vec::Vector{Int}
end

POMDPs.discount(m::TabularBenchPOMDP) = m.gamma
POMDPs.states(m::TabularBenchPOMDP) = m.states_vec
POMDPs.actions(m::TabularBenchPOMDP) = m.actions_vec
POMDPs.observations(m::TabularBenchPOMDP) = m.obs_vec

# Required by POMDPX/SARSOP writers that need deterministic indexing.
if isdefined(POMDPs, :stateindex)
    POMDPs.stateindex(::TabularBenchPOMDP, s::Int) = s
end
if isdefined(POMDPs, :actionindex)
    POMDPs.actionindex(::TabularBenchPOMDP, a::Int) = a
end
if isdefined(POMDPs, :obsindex)
    POMDPs.obsindex(::TabularBenchPOMDP, o::Int) = o
end

# POMDPs.jl API compatibility:
# - newer versions use `initialstate`
# - some older stacks use `initialstate_distribution`
if isdefined(POMDPs, :initialstate)
    POMDPs.initialstate(m::TabularBenchPOMDP) = _SparseCat(m.states_vec, m.init)
end
if isdefined(POMDPs, :initialstate_distribution)
    POMDPs.initialstate_distribution(m::TabularBenchPOMDP) = _SparseCat(m.states_vec, m.init)
end

function POMDPs.transition(m::TabularBenchPOMDP, s::Int, a::Int)
    return _SparseCat(m.states_vec, vec(m.T[a, s, :]))
end

function POMDPs.observation(m::TabularBenchPOMDP, a::Int, sp::Int)
    return _SparseCat(m.obs_vec, vec(m.O[a, sp, :]))
end

POMDPs.reward(m::TabularBenchPOMDP, s::Int, a::Int) = m.R[s, a]
POMDPs.reward(m::TabularBenchPOMDP, s::Int, a::Int, sp::Int) = m.R[s, a]

function parse_args(args::Vector{String})
    out = Dict{String, String}()
    i = 1
    while i <= length(args)
        key = args[i]
        if startswith(key, "--") && i < length(args)
            out[key] = args[i + 1]
            i += 2
        else
            i += 1
        end
    end
    return out
end

function parse_model(path::String)
    raw = JSON3.read(read(path, String))

    tr = raw["transition"]
    ob = raw["observation"]
    rw = raw["reward"]
    ib = raw["initial_belief"]

    n_actions = length(tr)
    n_states = length(tr[1])
    n_states_t = length(tr[1][1])
    n_obs = length(ob[1][1])

    if n_states != n_states_t
        error("Transition tensor must be square in state dimensions.")
    end

    T = zeros(Float64, n_actions, n_states, n_states)
    for a in 1:n_actions, s in 1:n_states, sp in 1:n_states
        T[a, s, sp] = Float64(tr[a][s][sp])
    end

    O = zeros(Float64, n_actions, n_states, n_obs)
    for a in 1:n_actions, sp in 1:n_states, o in 1:n_obs
        O[a, sp, o] = Float64(ob[a][sp][o])
    end

    R = zeros(Float64, n_states, n_actions)
    for s in 1:n_states, a in 1:n_actions
        R[s, a] = Float64(rw[s][a])
    end

    gamma = Float64(raw["gamma"])
    init = [Float64(x) for x in ib]

    # Defensive normalization to avoid floating-point drift from serialization.
    for a in 1:n_actions
        for s in 1:n_states
            row = vec(T[a, s, :])
            denom = sum(row)
            if denom > 0.0
                T[a, s, :] .= row ./ denom
            end
        end
    end

    for a in 1:n_actions
        for s in 1:n_states
            row = vec(O[a, s, :])
            denom = sum(row)
            if denom > 0.0
                O[a, s, :] .= row ./ denom
            end
        end
    end

    init_sum = sum(init)
    if init_sum > 0.0
        init ./= init_sum
    end

    return TabularBenchPOMDP(
        T,
        O,
        R,
        gamma,
        init,
        collect(1:n_states),
        collect(1:n_actions),
        collect(1:n_obs),
    )
end

function _is_number_vector(v)
    if !(v isa AbstractVector)
        return false
    end
    return all(x -> x isa Number, v)
end

function _extract_alpha(raw_alpha)
    if _is_number_vector(raw_alpha)
        return Float64.(collect(raw_alpha))
    end
    if hasproperty(raw_alpha, :alpha)
        return Float64.(collect(getproperty(raw_alpha, :alpha)))
    end
    if hasproperty(raw_alpha, :v)
        return Float64.(collect(getproperty(raw_alpha, :v)))
    end
    return nothing
end

function extract_policy(policy)
    if !hasproperty(policy, :alphas)
        error("SARSOP policy does not expose alpha vectors via `.alphas`.")
    end

    raw_alphas = getproperty(policy, :alphas)
    alpha_vectors = Vector{Vector{Float64}}()
    for raw in raw_alphas
        alpha = _extract_alpha(raw)
        if alpha === nothing
            error("Unsupported alpha-vector entry type in policy output.")
        end
        push!(alpha_vectors, alpha)
    end

    # Try standard action_map first.
    raw_actions = nothing
    if hasproperty(policy, :action_map)
        raw_actions = collect(getproperty(policy, :action_map))
    elseif hasproperty(policy, :actions)
        raw_actions = collect(getproperty(policy, :actions))
    end

    if raw_actions === nothing || length(raw_actions) != length(alpha_vectors)
        # Fallback: try action attached to each alpha vector entry.
        raw_actions = Vector{Any}(undef, length(raw_alphas))
        for i in eachindex(raw_alphas)
            entry = raw_alphas[i]
            if hasproperty(entry, :action)
                raw_actions[i] = getproperty(entry, :action)
            elseif hasproperty(entry, :a)
                raw_actions[i] = getproperty(entry, :a)
            else
                error("Could not extract action map from SARSOP policy.")
            end
        end
    end

    actions = Int[]
    for a in raw_actions
        push!(actions, Int(a))
    end

    # Convert 1-based Julia actions to 0-based Python actions.
    if minimum(actions) >= 1
        actions = [a - 1 for a in actions]
    end

    return alpha_vectors, actions
end

function main()
    args = parse_args(ARGS)
    model_json = get(args, "--model-json", "")
    output_json = get(args, "--output-json", "")
    timeout_sec = parse(Float64, get(args, "--timeout-sec", "120.0"))
    precision = parse(Float64, get(args, "--precision", "1e-3"))

    if isempty(model_json) || isempty(output_json)
        error("Usage: julia sarsop_bridge.jl --model-json <path> --output-json <path> [--timeout-sec <sec>] [--precision <eps>]")
    end

    model = parse_model(model_json)

    solver = SARSOPSolver(timeout=timeout_sec, precision=precision)
    policy = solve(solver, model)

    alphas, actions = extract_policy(policy)

    out = Dict(
        "solver" => "SARSOP.jl",
        "alpha_count" => length(alphas),
        "alphas" => alphas,
        "actions" => actions,
    )

    open(output_json, "w") do io
        JSON3.write(io, out)
    end
end

try
    main()
catch err
    msg = sprint(showerror, err, catch_backtrace())
    stderr_msg = Dict("error" => msg)
    if length(ARGS) >= 4
        # Best effort to write machine-readable error where Python expects it.
        parsed = parse_args(ARGS)
        out_path = get(parsed, "--output-json", "")
        if !isempty(out_path)
            try
                open(out_path, "w") do io
                    JSON3.write(io, stderr_msg)
                end
            catch
            end
        end
    end
    println(stderr, msg)
    exit(1)
end
