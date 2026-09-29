"""Section 2.1, p.3: T(a,b)=(a^Gamma+b^Gamma-1)^(1/Gamma)."""

from math import exp, expm1, isfinite, log, log1p


def _unit(value):
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Expected a finite value in [0, 1]")


def _parameter(parameter, positive):
    if not isfinite(parameter):
        raise ValueError("Parameter must be finite")
    if positive not in ("reject", "literal", "truncate"):
        raise ValueError("Unknown positive-parameter interpretation")
    if parameter > 0 and positive == "reject":
        raise ValueError("Section 2.1 requires Gamma < 0; select an explicit extension")


def inverse_generator(base, parameter, *, positive="reject"):
    """Invert a generator; never silently take abs() of a negative base."""
    _parameter(parameter, positive)
    if base <= 0:
        if parameter > 0 and (base == 0 or positive == "truncate"):
            return 0.0
        raise ValueError(f"SS power has non-positive base {base:g}")
    result = exp(log(base) / parameter)
    if not isfinite(result) or not 0 <= result <= 1 + 1e-12:
        raise ValueError(f"SS result outside [0,1]: {result:g}")
    return min(1.0, result)


def schweizer_sklar_t_norm(a, b, parameter, *, positive="reject"):
    """Printed negative-exponent convention; Gamma=0 uses its product limit.

    positive='literal' evaluates the untruncated p.4 expression where real;
    positive='truncate' explicitly extends it using max(0, base).
    """
    _unit(a)
    _unit(b)
    _parameter(parameter, positive)
    if parameter == 0:
        return a * b
    if parameter < 0:
        if min(a, b) == 0:
            return 0.0
        # Factor out the largest exponent to avoid overflow near zero inputs.
        u, v = sorted((parameter * log(a), parameter * log(b)), reverse=True)
        if u < 0.5:
            log_base = log1p(expm1(u) + expm1(v))
        else:
            log_base = u + log1p(exp(v - u) - exp(-u))
        return exp(log_base / parameter)
    return inverse_generator(a**parameter + b**parameter - 1, parameter, positive=positive)


def schweizer_sklar_t_conorm(a, b, parameter, *, positive="reject"):
    """Dual S(a,b)=1-T(1-a,1-b), Section 2.1."""
    _unit(a)
    _unit(b)
    return 1 - schweizer_sklar_t_norm(1 - a, 1 - b, parameter, positive=positive)


def repeated_norm(x, weight, parameter, *, plus=False, positive="reject"):
    """Scalar generator: 1+w(x^Gamma-1); printed variant w*x^Gamma+(w-1)."""
    _unit(x)
    _parameter(parameter, positive)
    if not isfinite(weight) or weight < 0:
        raise ValueError("Scalar weight must be finite and non-negative")
    if plus:
        if parameter == 0:
            raise ValueError("Printed +(w-1) law has no general Gamma=0 limit")
        if x == 0 and parameter < 0 and weight > 0:
            return 0.0
        if x == 0 and parameter < 0:
            raise ValueError("Printed law contains 0 * 0^Gamma")
        return inverse_generator(weight * x**parameter + weight - 1, parameter, positive=positive)
    if weight == 0:
        return 1.0
    if parameter == 0:
        return x**weight
    if x == 0 and parameter < 0:
        return 0.0
    return inverse_generator(1 + weight * expm1(parameter * log(x)) if x else 1-weight,
                             parameter, positive=positive)
