from math import cbrt
import pytest

from src.cff import CFFNumber
from src.operations import cff_add, cff_multiply, cff_scalar_mul, cff_power

A = CFFNumber(.2,.1,.4,.3)
B = CFFNumber(.3,.2,.5,.4)


def complement(x):
    return CFFNumber(x.nu_magnitude,x.nu_phase,x.mu_magnitude,x.mu_phase)


def test_literal_binary_laws_against_hand_formula():
    add = cff_add(A,B,-1)
    expected = (
        cbrt(1-1/(1/(1-.2**3)+1/(1-.3**3)-1)),
        cbrt(1-1/(1/(1-.1**3)+1/(1-.2**3)-1)),
        1/(1/.4+1/.5-1), 1/(1/.3+1/.4-1))
    assert add.components == pytest.approx(expected,abs=1e-4,rel=0)
    mult = cff_multiply(A,B,-1)
    assert mult.components == pytest.approx(complement(cff_add(complement(A),complement(B),-1)).components,
                                           abs=1e-4,rel=0)
    assert mult.mu_magnitude == pytest.approx(1/(1/.2+1/.3-1),abs=1e-4,rel=0)


@pytest.mark.parametrize("mode", ["literal","sign_corrected","fermatean"])
def test_scalar_unit_identity(mode):
    assert cff_scalar_mul(1,A,-.1,interpretation=mode).components == pytest.approx(A.components,abs=1e-4,rel=0)
    assert cff_power(A,1,-.1,interpretation=mode).components == pytest.approx(A.components,abs=1e-4,rel=0)


@pytest.mark.parametrize("mode", ["sign_corrected","fermatean"])
def test_corrected_scalar_matches_repeated_binary_operation(mode):
    assert cff_scalar_mul(2,A,-.1,interpretation=mode).components == pytest.approx(
        cff_add(A,A,-.1,interpretation=mode).components,abs=1e-4,rel=0)
    assert cff_power(A,2,-.1,interpretation=mode).components == pytest.approx(
        cff_multiply(A,A,-.1,interpretation=mode).components,abs=1e-4,rel=0)


def test_printed_scalar_sign_has_a_concrete_counterexample():
    # Def.3.1 gives 2*p^Gamma+1 instead of the binary law's 2*p^Gamma-1.
    scaled = cff_scalar_mul(2,A,-1)
    doubled = cff_add(A,A,-1)
    assert scaled.nu_magnitude == pytest.approx(1/(2/.4+1),abs=1e-4,rel=0)
    assert abs(scaled.nu_magnitude-doubled.nu_magnitude) > 1e-4
    with pytest.raises(ValueError):
        cff_scalar_mul(.5,A,-.1)


def test_literal_power_preserves_printed_plus_sign():
    result = cff_power(A,2,-1)
    assert result.mu_magnitude == pytest.approx(1/(2/.2+1),abs=1e-4,rel=0)
    assert result.nu_magnitude == pytest.approx(cbrt(1-1/(2/(1-.4**3)+1)),abs=1e-4,rel=0)


@pytest.mark.parametrize("fn", [lambda: cff_scalar_mul(-1,A,-.1),
    lambda: cff_power(A,float("nan"),-.1), lambda: cff_add(A,B,-.1,interpretation="typo")])
def test_invalid_options(fn):
    with pytest.raises(ValueError):
        fn()


def test_fermatean_cubed_norm_is_an_explicit_different_interpretation():
    x = cff_add(A,B,-1,interpretation="fermatean")
    assert x.nu_magnitude == pytest.approx(cbrt(1/(1/.4**3+1/.5**3-1)),abs=1e-4,rel=0)
    assert abs(x.nu_magnitude-cff_add(A,B,-1).nu_magnitude) > 1e-4
