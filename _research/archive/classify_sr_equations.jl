#=========================================================
MODULE TO CLASSIFY EQUATIONS OUTPUT BY SYMBOLIC REGRESSION
==========================================================#  

module SR_equation_classification

export classify_eqn

#============================================================
REGEX PATTERNS
=============================================================#

# Floating point number pattern (positive, negative, scientific notation)
const NUM = raw"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
println("number pattern: ", NUM)

#============================================================
CORRECT PATTERNS
=============================================================#

# a * exp(-b * x1)   — constant times exp of negative-times-x1
const PAT_STANDARD = Regex(
    "^\\(?($NUM)\\s*[\\*×]\\s*exp\\(\\s*($NUM)\\s*[\\*×]\\s*x1\\s*\\)\\)?\$"
)

# a * exp(x1 * -b)   — same but operand order flipped
const PAT_STANDARD_FLIP = Regex(
    "^\\(?($NUM)\\s*[\\*×]\\s*exp\\(\\s*x1\\s*[\\*×]\\s*($NUM)\\s*\\)\\)?\$"
)

# exp(x1 * -b) * a  — same but operand order flipped
const PAT_STANDARD_FLIP2 = Regex(
    "^exp\\(\\s*x1\\s*[\\*×]\\s*($NUM)\\s*\\)\\s*[\\*×]\\s*($NUM)\$"
)

# exp(-b * x1) * a  — constant times exp of negative-times-x1
const PAT_STANDARD2 = Regex(
    "^exp\\(\\s*($NUM)\\s*[\\*×]\\s*x1\\s*\\)\\s*[\\*×]\\s*($NUM)\$"
)

# a * exp(x1 / -c)   — division form
const PAT_DIV = Regex(
    "^\\(?($NUM)\\s*[\\*×]\\s*exp\\(\\s*x1\\s*/\\s*($NUM)\\s*\\)\\)?\$"
)

# a * exp(-x1 / c)   — negated division form
const PAT_NEGDIV = Regex(
    "^\\(?($NUM)\\s*[\\*×]\\s*exp\\(\\s*-\\s*x1\\s*/\\s*($NUM)\\s*\\)\\)?\$"
)

# exp(x1 / -c) * a   — division form
const PAT_DIV_FLIP = Regex(
    "^exp\\(\\s*x1\\s*/\\s*($NUM)\\s*\\)\\s*[\\*×]\\s*($NUM)\$"
)

# a / exp(x1 / c)   — reciprocal exponential form
const PAT_RECIP_EXP = Regex(
    "^\\(?($NUM)\\s*/\\s*exp\\(\\s*x1\\s*/\\s*($NUM)\\s*\\)\\)?\$"
)

# a / exp(x1 * c)   — reciprocal exponential form with multiplication inside exp
const PAT_RECIP_EXP_MUL = Regex(
    "^\\(?($NUM)\\s*/\\s*exp\\(\\s*x1\\s*[\\*×]\\s*($NUM)\\s*\\)\\)?\$"
)

# a / exp(c * x1)  — reciprocal exponential form with multiplication inside exp, flipped order
const PAT_RECIP_EXP_MUL2 = Regex(
    "^\\(?($NUM)\\s*/\\s*exp\\(\\s*($NUM)\\s*[\\*×]\\s*x1\\s*\\)\\)?\$"
)

# a * exp(b * x1) / exp(c * x1)
const PAT_EXP_RATIO1 = Regex(
    "^($NUM)\\s*[\\*×]\\s*exp\\(\\s*($NUM)\\s*[\\*×]\\s*x1\\s*\\)\\s*/\\s*exp\\(\\s*($NUM)\\s*[\\*×]\\s*x1\\s*\\)\$"
)
# a * exp(x1 * b) / exp(c * x1)
const PAT_EXP_RATIO2 = Regex(
    "^($NUM)\\s*[\\*×]\\s*exp\\(\\s*x1\\s*[\\*×]\\s*($NUM)\\s*\\)\\s*/\\s*exp\\(\\s*($NUM)\\s*[\\*×]\\s*x1\\s*\\)\$"
)
# a * exp(b * x1) / exp(x1 * c)
const PAT_EXP_RATIO3 = Regex(
    "^($NUM)\\s*[\\*×]\\s*exp\\(\\s*($NUM)\\s*[\\*×]\\s*x1\\s*\\)\\s*/\\s*exp\\(\\s*x1\\s*[\\*×]\\s*($NUM)\\s*\\)\$"
)
# a * exp(x1 * b) / exp(x1 * c)
const PAT_EXP_RATIO4 = Regex(
    "^($NUM)\\s*[\\*×]\\s*exp\\(\\s*x1\\s*[\\*×]\\s*($NUM)\\s*\\)\\s*/\\s*exp\\(\\s*x1\\s*[\\*×]\\s*($NUM)\\s*\\)\$"
)
# exp(b * x1) / exp(c * x1) * a  — coefficient on right
const PAT_EXP_RATIO5 = Regex(
    "^exp\\(\\s*($NUM)\\s*[\\*×]\\s*x1\\s*\\)\\s*/\\s*exp\\(\\s*($NUM)\\s*[\\*×]\\s*x1\\s*\\)\\s*[\\*×]\\s*($NUM)\$"
)
# exp(x1 * b) / exp(x1 * c) * a
const PAT_EXP_RATIO6 = Regex(
    "^exp\\(\\s*x1\\s*[\\*×]\\s*($NUM)\\s*\\)\\s*/\\s*exp\\(\\s*x1\\s*[\\*×]\\s*($NUM)\\s*\\)\\s*[\\*×]\\s*($NUM)\$"
)
# (exp(b * x1) / exp(c * x1)) * a  — with outer parentheses
const PAT_EXP_RATIO7 = Regex(
    "^\\(exp\\(\\s*($NUM)\\s*[\\*×]\\s*x1\\s*\\)\\s*/\\s*exp\\(\\s*($NUM)\\s*[\\*×]\\s*x1\\s*\\)\\)\\s*[\\*×]\\s*($NUM)\$"
)
# (exp(x1 * b) / exp(x1 * c)) * a
const PAT_EXP_RATIO8 = Regex(
    "^\\(exp\\(\\s*x1\\s*[\\*×]\\s*($NUM)\\s*\\)\\s*/\\s*exp\\(\\s*x1\\s*[\\*×]\\s*($NUM)\\s*\\)\\)\\s*[\\*×]\\s*($NUM)\$"
)

#============================================================
INCORRECT PATTERNS
=============================================================#

# Nested exp
const PAT_NESTED = r"exp\s*\(\s*exp"

#============================================================
EXP WITH ADDITIVE CONSTANT IN EXPONENT
=============================================================#

# exp(a + b*x1)
const PAT_EXP_SUM1 = Regex(
    "^exp\\(\\s*($NUM)\\s*\\+\\s*($NUM)\\s*[\\*×]\\s*x1\\s*\\)\$"
)
# exp(a + x1*b)
const PAT_EXP_SUM2 = Regex(
    "^exp\\(\\s*($NUM)\\s*\\+\\s*x1\\s*[\\*×]\\s*($NUM)\\s*\\)\$"
)
# exp(b*x1 + a)
const PAT_EXP_SUM3 = Regex(
    "^exp\\(\\s*($NUM)\\s*[\\*×]\\s*x1\\s*\\+\\s*($NUM)\\s*\\)\$"
)
# exp(x1*b + a)
const PAT_EXP_SUM4 = Regex(
    "^exp\\(\\s*x1\\s*[\\*×]\\s*($NUM)\\s*\\+\\s*($NUM)\\s*\\)\$"
)
# exp(a - b*x1)
const PAT_EXP_SUM5 = Regex(
    "^exp\\(\\s*($NUM)\\s*-\\s*($NUM)\\s*[\\*×]\\s*x1\\s*\\)\$"
)
# exp(a - x1*b)
const PAT_EXP_SUM6 = Regex(
    "^exp\\(\\s*($NUM)\\s*-\\s*x1\\s*[\\*×]\\s*($NUM)\\s*\\)\$"
)
# exp(b*x1 - a)
const PAT_EXP_SUM7 = Regex(
    "^exp\\(\\s*($NUM)\\s*[\\*×]\\s*x1\\s*-\\s*($NUM)\\s*\\)\$"
)
# exp(x1*b - a)
const PAT_EXP_SUM8 = Regex(
    "^exp\\(\\s*x1\\s*[\\*×]\\s*($NUM)\\s*-\\s*($NUM)\\s*\\)\$"
)

# exp(a + b*x1) / exp(c + d*x1)
const PAT_EXP_SUM_RATIO1 = Regex(
    "^exp\\(\\s*($NUM)\\s*\\+\\s*($NUM)\\s*[\\*×]\\s*x1\\s*\\)\\s*/\\s*exp\\(\\s*($NUM)\\s*\\+\\s*($NUM)\\s*[\\*×]\\s*x1\\s*\\)\$"
)
# exp(a + x1*b) / exp(c + x1*d)
const PAT_EXP_SUM_RATIO2 = Regex(
    "^exp\\(\\s*($NUM)\\s*\\+\\s*x1\\s*[\\*×]\\s*($NUM)\\s*\\)\\s*/\\s*exp\\(\\s*($NUM)\\s*\\+\\s*x1\\s*[\\*×]\\s*($NUM)\\s*\\)\$"
)
# exp(b*x1 + a) / exp(d*x1 + c)
const PAT_EXP_SUM_RATIO3 = Regex(
    "^exp\\(\\s*($NUM)\\s*[\\*×]\\s*x1\\s*\\+\\s*($NUM)\\s*\\)\\s*/\\s*exp\\(\\s*($NUM)\\s*[\\*×]\\s*x1\\s*\\+\\s*($NUM)\\s*\\)\$"
)
# exp(x1*b + a) / exp(x1*d + c)
const PAT_EXP_SUM_RATIO4 = Regex(
    "^exp\\(\\s*x1\\s*[\\*×]\\s*($NUM)\\s*\\+\\s*($NUM)\\s*\\)\\s*/\\s*exp\\(\\s*x1\\s*[\\*×]\\s*($NUM)\\s*\\+\\s*($NUM)\\s*\\)\$"
)

# Mixed operand orders — numerator flipped vs denominator
# exp(a + b*x1) / exp(d*x1 + c)
const PAT_EXP_SUM_RATIO5 = Regex(
    "^exp\\(\\s*($NUM)\\s*\\+\\s*($NUM)\\s*[\\*×]\\s*x1\\s*\\)\\s*/\\s*exp\\(\\s*($NUM)\\s*[\\*×]\\s*x1\\s*\\+\\s*($NUM)\\s*\\)\$"
)
# exp(b*x1 + a) / exp(c + d*x1)
const PAT_EXP_SUM_RATIO6 = Regex(
    "^exp\\(\\s*($NUM)\\s*[\\*×]\\s*x1\\s*\\+\\s*($NUM)\\s*\\)\\s*/\\s*exp\\(\\s*($NUM)\\s*\\+\\s*($NUM)\\s*[\\*×]\\s*x1\\s*\\)\$"
)

# With subtraction in exponents
# exp(a - b*x1) / exp(c - d*x1)
const PAT_EXP_SUM_RATIO7 = Regex(
    "^exp\\(\\s*($NUM)\\s*-\\s*($NUM)\\s*[\\*×]\\s*x1\\s*\\)\\s*/\\s*exp\\(\\s*($NUM)\\s*-\\s*($NUM)\\s*[\\*×]\\s*x1\\s*\\)\$"
)
# exp(a - x1*b) / exp(c - x1*d)
const PAT_EXP_SUM_RATIO8 = Regex(
    "^exp\\(\\s*($NUM)\\s*-\\s*x1\\s*[\\*×]\\s*($NUM)\\s*\\)\\s*/\\s*exp\\(\\s*($NUM)\\s*-\\s*x1\\s*[\\*×]\\s*($NUM)\\s*\\)\$"
)

#============================================================
CLASSIFICATION FUNCTION
============================================================#

function classify_eqn(equation::String)
    eqn = replace(equation, " " => "")  # Remove all whitespace for easier matching
    println("Classifying equation: ", eqn)
    
    # CORRECT FORMS
    
    if occursin(PAT_STANDARD, eqn)
        return :correct, "Standard form: a * exp(-b * x1)"

    elseif occursin(PAT_STANDARD_FLIP, eqn)
        return :correct, "Standard form (flipped): a * exp(x1 * -b)"

    elseif occursin(PAT_STANDARD2, eqn)
        return :correct, "Standard form: exp(-b * x1) * a"

    elseif occursin(PAT_STANDARD_FLIP2, eqn)
        return :correct, "Standard form (flipped): exp(x1 * -b) * a"

    elseif occursin(PAT_DIV, eqn)
        return :correct, "Division form: a * exp(x1 / -c)"

    elseif occursin(PAT_NEGDIV, eqn)
        return :correct, "Negated division form: a * exp(-x1 / c)"

    elseif occursin(PAT_RECIP_EXP, eqn)
        return :correct, "Reciprocal exponential form: a / exp(x1 / c)"

    elseif occursin(PAT_RECIP_EXP_MUL, eqn)
        return :correct, "Reciprocal exponential form: a / exp(x1 * c)"

    elseif occursin(PAT_RECIP_EXP_MUL2, eqn)
        return :correct, "Reciprocal exponential form: a / exp(c * x1)"

    elseif occursin(PAT_DIV_FLIP, eqn)
        return :correct, "Exponential division form: exp(x1 / c) * a"

    elseif occursin(PAT_EXP_RATIO1, eqn)
        return :correct, "Ratio of exponentials form 1: a * exp(b * x1) / exp(c * x1)"

    elseif occursin(PAT_EXP_RATIO2, eqn)
        return :correct, "Ratio of exponentials form 2: a * exp(x1 * b) / exp(c * x1)"

    elseif occursin(PAT_EXP_RATIO3, eqn)
        return :correct, "Ratio of exponentials form 3: a * exp(b * x1) / exp(x1 * c)"

    elseif occursin(PAT_EXP_RATIO4, eqn)
        return :correct, "Ratio of exponentials form 4: a * exp(x1 * b) / exp(x1 * c)"

    elseif occursin(PAT_EXP_RATIO5, eqn)
        return :correct, "Ratio of exponentials form 5: exp(b * x1) / exp(c * x1) * a"

    elseif occursin(PAT_EXP_RATIO6, eqn)
        return :correct, "Ratio of exponentials form 6: exp(x1 * b) / exp(x1 * c) * a"

    elseif occursin(PAT_EXP_RATIO7, eqn)
        return :correct, "Ratio of exponentials form 7: (exp(b * x1) / exp(c * x1)) * a"

    elseif occursin(PAT_EXP_RATIO8, eqn)
        return :correct, "Ratio of exponentials form 8: (exp(x1 * b) / exp(x1 * c)) * a"

    # INCORRECT FORMS

    elseif occursin(PAT_NESTED, eqn)
        return :nested_exp, "Nested exp"
    
    elseif !occursin("x1", eqn) && occursin(r"^[0-9eE\.\+\-\s\*\/\(\)]+$", eqn)
        return :constant, "Constant"

    elseif occursin(PAT_EXP_SUM1, eqn)
        return :additive_constant_in_exp, "Exp sum form: exp(a + b*x1)"
    elseif occursin(PAT_EXP_SUM2, eqn)
        return :additive_constant_in_exp, "Exp sum form: exp(a + x1*b)"
    elseif occursin(PAT_EXP_SUM3, eqn)
        return :additive_constant_in_exp, "Exp sum form: exp(b*x1 + a)"
    elseif occursin(PAT_EXP_SUM4, eqn)
        return :additive_constant_in_exp, "Exp sum form: exp(x1*b + a)"
    elseif occursin(PAT_EXP_SUM5, eqn)
        return :additive_constant_in_exp, "Exp sum form: exp(a - b*x1)"
    elseif occursin(PAT_EXP_SUM6, eqn)
        return :additive_constant_in_exp, "Exp sum form: exp(a - x1*b)"
    elseif occursin(PAT_EXP_SUM7, eqn)
        return :additive_constant_in_exp, "Exp sum form: exp(b*x1 - a)"
    elseif occursin(PAT_EXP_SUM8, eqn)
        return :additive_constant_in_exp, "Exp sum form: exp(x1*b - a)"
    elseif occursin(PAT_EXP_SUM_RATIO1, eqn)
        return :additive_constant_in_exp, "Exp sum ratio: exp(a+b*x1)/exp(c+d*x1)"
    elseif occursin(PAT_EXP_SUM_RATIO2, eqn)
        return :additive_constant_in_exp, "Exp sum ratio: exp(a+x1*b)/exp(c+x1*d)"
    elseif occursin(PAT_EXP_SUM_RATIO3, eqn)
        return :additive_constant_in_exp, "Exp sum ratio: exp(b*x1+a)/exp(d*x1+c)"
    elseif occursin(PAT_EXP_SUM_RATIO4, eqn)
        return :additive_constant_in_exp, "Exp sum ratio: exp(x1*b+a)/exp(x1*d+c)"
    elseif occursin(PAT_EXP_SUM_RATIO5, eqn)
        return :additive_constant_in_exp, "Exp sum ratio: exp(a+b*x1)/exp(d*x1+c)"
    elseif occursin(PAT_EXP_SUM_RATIO6, eqn)
        return :additive_constant_in_exp, "Exp sum ratio: exp(b*x1+a)/exp(c+d*x1)"
    elseif occursin(PAT_EXP_SUM_RATIO7, eqn)
        return :additive_constant_in_exp, "Exp sum ratio: exp(a-b*x1)/exp(c-d*x1)"
    elseif occursin(PAT_EXP_SUM_RATIO8, eqn)
        return :additive_constant_in_exp, "Exp sum ratio: exp(a-x1*b)/exp(c-x1*d)"
    else
        return :unclassified, "Unclassified"
    end

end

end