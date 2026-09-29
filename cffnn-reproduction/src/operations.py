"""Definition 3.1, p.4, with explicit, non-interchangeable interpretations."""

from math import cbrt

from .cff import CFFNumber
from .schweizer_sklar import schweizer_sklar_t_norm as tn
from .schweizer_sklar import schweizer_sklar_t_conorm as tc
from .schweizer_sklar import repeated_norm

INTERPRETATIONS = ("literal", "sign_corrected", "fermatean")


def _check(interpretation):
    if interpretation not in INTERPRETATIONS:
        raise ValueError(f"Choose one of {INTERPRETATIONS}")


def _norm(a, b, parameter, interpretation, positive):
    if interpretation == "fermatean":
        return cbrt(tn(a**3, b**3, parameter, positive=positive))
    return tn(a, b, parameter, positive=positive)


def _conorm(a, b, parameter, positive):
    return cbrt(tc(a**3, b**3, parameter, positive=positive))


def cff_add(a, b, parameter, *, interpretation="literal", positive="reject"):
    """Def.3.1 first law. Printed nu branch has no cubes/cube root."""
    _check(interpretation)
    return CFFNumber(*[
        _conorm(x, y, parameter, positive) if i < 2
        else _norm(x, y, parameter, interpretation, positive)
        for i, (x, y) in enumerate(zip(a.components, b.components))
    ])


def cff_multiply(a, b, parameter, *, interpretation="literal", positive="reject"):
    """Def.3.1 second law; dual of cff_add."""
    _check(interpretation)
    return CFFNumber(*[
        _norm(x, y, parameter, interpretation, positive) if i < 2
        else _conorm(x, y, parameter, positive)
        for i, (x, y) in enumerate(zip(a.components, b.components))
    ])


def _scalar(weight, x, parameter, power, interpretation, positive):
    _check(interpretation)
    out = []
    for i, v in enumerate(x.components):
        conorm_branch = (i >= 2) if power else (i < 2)
        plus = interpretation == "literal"
        if conorm_branch:
            out.append(cbrt(1 - repeated_norm(1-v**3, weight, parameter,
                                              plus=plus, positive=positive)))
        elif interpretation == "fermatean":
            out.append(cbrt(repeated_norm(v**3, weight, parameter, positive=positive)))
        else:
            out.append(repeated_norm(v, weight, parameter, plus=plus, positive=positive))
    return CFFNumber(*out)


def cff_scalar_mul(weight, x, parameter, *, interpretation="literal", positive="reject"):
    """Def.3.1 third law: literal preserves printed +(n-1), even if invalid."""
    return _scalar(weight, x, parameter, False, interpretation, positive)


def cff_power(x, weight, parameter, *, interpretation="literal", positive="reject"):
    """Def.3.1 fourth law, with the same explicit sign/cube hypotheses."""
    return _scalar(weight, x, parameter, True, interpretation, positive)
