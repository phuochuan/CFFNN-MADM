"""Sections 4.1--4.3; definitions and conjectural repairs kept explicit."""

from functools import reduce
from math import cbrt, isclose, isfinite, prod

from .cff import CFFNumber
from .operations import cff_add, cff_scalar_mul
from .schweizer_sklar import inverse_generator


def validate_weights(values, weights):
    values, weights = tuple(values), tuple(weights)
    if not values or len(values) != len(weights):
        raise ValueError("Non-empty values and weights must have equal lengths")
    if any(not isfinite(w) or w < 0 or w > 1 for w in weights):
        raise ValueError("Weights must be finite and in [0,1]")
    if not isclose(sum(weights), 1, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Weights must sum to 1")
    return values, weights


def cffsstwa(values, weights, parameter, *, interpretation="literal", positive="reject"):
    """Def.4.1.1: direct fold of the scalar and addition laws, without rounding."""
    values, weights = validate_weights(values, weights)
    options = dict(interpretation=interpretation, positive=positive)
    terms = [cff_scalar_mul(w, v, parameter, **options) for v, w in zip(values, weights)]
    return reduce(lambda a, b: cff_add(a, b, parameter, **options), terms)


def cffsstowa(values, weights, parameter, *, key=None, **options):
    """Def.4.2.1 prints the same fold as WA, without a permutation.

    With an explicit key, evaluate the conventional descending ordered variant
    as a hypothesis. No undocumented real score is chosen automatically.
    """
    values, weights = validate_weights(values, weights)
    if key is not None:
        values = tuple(sorted(values, key=key, reverse=True))
    return cffsstwa(values, weights, parameter, **options)


def cffssthwa(values, weights, parameter, *, order_weights=None, key=None, **options):
    """Def.4.3.1 again prints WA. Conventional hybrid is opt-in only.

    Hybrid hypothesis: scale inputs by n*w_i, sort using caller's key, then
    aggregate with a second explicitly supplied positional weight vector.
    Neither that second vector nor the sorting rule is supplied by the paper.
    """
    values, weights = validate_weights(values, weights)
    if order_weights is None:
        if key is not None:
            raise ValueError("Hybrid ordering requires explicit order_weights")
        return cffsstwa(values, weights, parameter, **options)
    if key is None:
        raise ValueError("Hybrid interpretation requires an explicit ordering key")
    _, order_weights = validate_weights(values, order_weights)
    scaled = [cff_scalar_mul(len(values)*w, v, parameter, **options)
              for v, w in zip(values, weights)]
    return cffsstowa(scaled, order_weights, parameter, key=key, **options)


def printed_hwa_product(values, *, rho, theta, indexing="repeated_first"):
    """Theorem 4.3.2 p.6, product expression, separate from Def.4.3.1.

    Paper leaves rho/theta mapping undefined and repeats L_1,M_1 inside products.
    indexing='indexed' explicitly hypothesizes L_j,M_j instead. theta is a
    common exponent, NOT a vector of criterion weights. No domain clipping.
    """
    values = tuple(values)
    if not values or not isfinite(rho) or rho <= 0 or not isfinite(theta) or theta == 0:
        raise ValueError("Require nonempty values, rho>0 and finite theta!=0")
    if indexing not in ("repeated_first", "indexed"):
        raise ValueError("Unknown indexing interpretation")
    if indexing == "repeated_first":
        values = (values[0],) * len(values)
    out = []
    for i in range(4):
        if i < 2:
            factors = [inverse_generator(rho*(1-v.components[i]**3)**theta + rho-1,
                                         theta, positive="literal") for v in values]
            out.append(cbrt(1-prod(factors)))
        else:
            base = rho*prod(v.components[i]**theta for v in values) + rho-1
            out.append(inverse_generator(base, theta, positive="literal"))
    return CFFNumber(*out)
