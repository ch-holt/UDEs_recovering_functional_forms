#========================================================
CLASSIFY SR EQUATIONS BY MECHANISM (beta_exp ground truth a * exp(-b * x))

Works on the raw equation strings SR saves (r.equation_strings), which are
fully parenthesised with explicit operators, e.g.
    "0.2404 / exp(x1 / 0.1064)"
    "exp(x1 * -5.53) * 0.542 + 0.00231"
Meta.parse turns each string into a Julia expression tree (Expr), which is
walked instead of matched with regex, so the order SR writes the operands in
doesn't matter.

Key idea: any subtree without x1 in it is just a number. So a "ground-truth
term" is a product/quotient of x-free constants and exp(g), with g linear in
x1 — however SR has arranged it:
    a * exp(-b*x1), exp(x1 * -b) * a, a / exp(x1 / c), exp(c - b*x1), ...

Categories (of the whole equation):
    :exact              one term, and it is a ground-truth term
    :gt_plus_correction several additive terms, exactly one a ground-truth term
                        (e.g. 0.002 + 0.54*exp(-5.5*x1), or ... + 0.099*x1)
    :perturbed_exponent has an a*exp(g) term with g a polynomial of degree >= 2
                        (e.g. 0.54*exp(-5.4*x1 - 0.77*x1^2 + 2*x1^3))
    :other              anything else (nested exp, x1 in a denominator, ...)

Parameters: for the ground-truth term T(x) = a*exp(-b*x), a = T(0) and
b = -log(T(1)/a), read off numerically so they don't depend on how SR wrote
it. params_ok = a and b both within rtol (default 20%) of the truth.

Only the exp form is implemented. For beta_mixed / beta_rational, add a
matcher like exp_term_degree and a parameter extractor like exp_params;
the parsing, evaluation and additive split are shared.

Run run_classifier_tests() to check it on known equations.
=========================================================#

module SREquationClassifier

using Test

export parse_equation, has_x, eval_at, poly_degree, summands, exp_term_degree, exp_params,
       classify_structure, classify_equation, run_classifier_tests

# The input variable SR uses (I/N here)
const XVAR = :x1

parse_equation(equation::AbstractString) = Meta.parse(equation)

#==============================================================
# Basic tree helpers
==============================================================#

# Does x1 appear anywhere in this subtree? (args[1] of a call is the operator, so skip it)
has_x(ex::Number) = false
has_x(ex::Symbol) = ex == XVAR
has_x(ex::Expr)   = any(has_x, ex.args[2:end])

# Evaluate the expression at x1 = x (only the operators SR was given: + - * / exp)
eval_at(ex::Number, x) = Float64(ex)
eval_at(ex::Symbol, x) = ex == XVAR ? Float64(x) : error("Unknown variable $(ex)")
function eval_at(ex::Expr, x)
    ex.head == :call || error("Unexpected expression $(ex)")
    op   = ex.args[1]
    args = [eval_at(a, x) for a in ex.args[2:end]]
    op == :+   && return sum(args)
    op == :-   && return length(args) == 1 ? -args[1] : args[1] - args[2]
    op == :*   && return prod(args)
    op == :/   && return args[1] / args[2]
    op == :exp && return exp(args[1])
    error("Unsupported operator $(op)")
end

# Polynomial degree in x1, or nothing if it isn't a polynomial
# (x-free subtrees, including exp(constant), count as constants: degree 0)
poly_degree(ex::Number) = 0
poly_degree(ex::Symbol) = ex == XVAR ? 1 : nothing
function poly_degree(ex::Expr)
    has_x(ex) || return 0
    op   = ex.args[1]
    args = ex.args[2:end]
    degs = [poly_degree(a) for a in args]
    any(isnothing, degs) && return nothing
    (op == :+ || op == :-) && return maximum(degs)
    op == :* && return sum(degs)
    # dividing by a constant keeps a polynomial; dividing by something with x1 doesn't
    op == :/ && return has_x(args[2]) ? nothing : degs[1]
    return nothing   # e.g. exp(x1)
end

#==============================================================
# Split into additive terms, and terms into multiplicative factors
==============================================================#

# a + b - c  ->  [a, b, -c]
function summands(ex, out = Any[])
    if ex isa Expr && ex.head == :call && ex.args[1] == :+
        foreach(a -> summands(a, out), ex.args[2:end])
    elseif ex isa Expr && ex.head == :call && ex.args[1] == :- && length(ex.args) == 3
        summands(ex.args[2], out)
        push!(out, Expr(:call, :-, ex.args[3]))
    else
        push!(out, ex)
    end
    return out
end

# a * b / c  ->  [a, b, c] (powers dropped: only which factors contain x1 matters here)
function factors(ex, out = Any[])
    if ex isa Expr && ex.head == :call && ex.args[1] == :*
        foreach(a -> factors(a, out), ex.args[2:end])
    elseif ex isa Expr && ex.head == :call && ex.args[1] == :/
        factors(ex.args[2], out)
        factors(ex.args[3], out)
    elseif ex isa Expr && ex.head == :call && ex.args[1] == :- && length(ex.args) == 2
        factors(ex.args[2], out)   # unary minus is just a constant factor of -1
    else
        push!(out, ex)
    end
    return out
end

# If a term is (x-free constants) x exp(g1) x exp(g2) ... with every g a polynomial
# in x1, return the highest degree of the g's (1 = ground-truth form, >= 2 = perturbed
# exponent). Return nothing for anything else (x1 outside an exp, exp of exp, ...).
function exp_term_degree(term)
    x_factors = filter(has_x, factors(term))
    isempty(x_factors) && return nothing
    degs = Int[]
    for f in x_factors
        (f isa Expr && f.head == :call && f.args[1] == :exp) || return nothing
        d = poly_degree(f.args[2])
        isnothing(d) && return nothing
        push!(degs, d)
    end
    return maximum(degs)
end

# For a ground-truth term T(x) = a * exp(-b x): a = T(0), b = -log(T(1) / a)
function exp_params(term)
    a  = eval_at(term, 0.0)
    t1 = eval_at(term, 1.0)
    (isfinite(a) && isfinite(t1) && a > 0 && t1 > 0) || return (NaN, NaN)
    return (a, -log(t1 / a))
end

#==============================================================
# Classify
==============================================================#

# Returns (category, ground-truth term or nothing)
function classify_structure(ex)
    terms  = summands(ex)
    xterms = filter(has_x, terms)
    degs   = [exp_term_degree(t) for t in xterms]
    gt_idx = findall(d -> d == 1, degs)
    if length(terms) == 1 && length(gt_idx) == 1
        return (:exact, xterms[only(gt_idx)])
    elseif length(gt_idx) == 1
        return (:gt_plus_correction, xterms[only(gt_idx)])
    elseif any(d -> !isnothing(d) && d >= 2, degs)
        return (:perturbed_exponent, nothing)
    else
        return (:other, nothing)
    end
end

function classify_equation(equation::AbstractString, a_true, b_true; rtol = 0.2)
    structure, gt_term = classify_structure(parse_equation(equation))
    a_hat, b_hat = isnothing(gt_term) ? (NaN, NaN) : exp_params(gt_term)
    params_ok = isapprox(a_hat, a_true; rtol) && isapprox(b_hat, b_true; rtol)
    return (; structure, a_hat, b_hat, params_ok)
end

#==============================================================
# Tests: the screenshot equations (truth 0.533 * exp(-5.41 x)), the same
# form written the other ways SR writes it, and real strings from the fronts
==============================================================#

function run_classifier_tests()
    a, b = 0.533, 5.41
    cases = [
        # screenshot (written in SR's raw format)
        ("0.542 * exp(-5.53 * x1)",                                         :exact,              true),
        ("0.00231 + 0.538 * exp(-5.54 * x1)",                               :gt_plus_correction, true),
        ("0.537 * exp(-5.87 * x1) + 0.099 * x1",                            :gt_plus_correction, true),
        ("0.538 * exp(((-5.39 * x1) - (0.771 * (x1 * x1))) + (2.0 * ((x1 * x1) * x1)))", :perturbed_exponent, false),
        ("exp((exp(2.03 + (x1 + (x1 * x1))) * (0.137 + x1)) / ((-0.217 * exp(2.03 + (x1 + (x1 * x1)))) + x1))", :other, false),
        # same ground truth, rearranged
        ("exp(x1 * -5.53) * 0.542",                                         :exact,              true),
        ("exp(-0.612 - (5.53 * x1))",                                       :exact,              true),
        ("exp(-5.53 * x1) / 1.845",                                         :exact,              true),
        ("0.542 / exp(x1 / 0.1808)",                                        :exact,              true),
        ("exp(x1 * -3.0) * (0.542 / exp(x1 * 2.53))",                       :exact,              true),
        # right form, wrong parameters (nearly constant)
        ("exp(x1 * -2.2269695011946026e-7) * 0.5943123359333966",           :exact,              false),
        # real non-ground-truth shapes from the fronts
        ("exp(exp(x1) / -0.44123069469925275)",                             :other,              false),
        ("0.24375280496202592 / (exp(x1 / 0.10859520378671643) + x1)",      :other,              false),
        ("0.05691653304107486 / exp(x1)",                                   :exact,              false),
        ("x1",                                                              :other,              false),
        ("0.02827512737076551",                                             :other,              false),
    ]
    @testset "SR equation classifier" begin
        for (eq, structure, ok) in cases
            r = classify_equation(eq, a, b)
            @test r.structure == structure
            @test r.params_ok == ok
        end
        # parameters read off a rearranged form
        r = classify_equation("exp(-0.612 - (5.53 * x1))", a, b)
        @test r.a_hat ≈ exp(-0.612)
        @test r.b_hat ≈ 5.53
    end
end

end # module
