import pytest

from src.schweizer_sklar import schweizer_sklar_t_norm as tn
from src.schweizer_sklar import schweizer_sklar_t_conorm as tc


@pytest.mark.parametrize("p", [-10., -1., -.1, 0.])
@pytest.mark.parametrize("x", [0., .2, .8, 1.])
def test_boundary_values(p,x):
    assert tn(x,1,p) == pytest.approx(x,abs=1e-4,rel=0)
    assert tn(x,0,p) == pytest.approx(0,abs=1e-4,rel=0)
    assert tc(x,0,p) == pytest.approx(x,abs=1e-4,rel=0)
    assert tc(x,1,p) == pytest.approx(1,abs=1e-4,rel=0)


def test_independent_closed_form_and_product_limit():
    assert tn(.4,.7,-1) == pytest.approx(1/(1/.4+1/.7-1),abs=1e-4,rel=0)
    assert tc(.4,.7,-1) == pytest.approx(1-1/(1/.6+1/.3-1),abs=1e-4,rel=0)
    for p in (0,-1e-10):
        assert tn(.4,.7,p) == pytest.approx(.28,abs=1e-4,rel=0)
        assert tc(.4,.7,p) == pytest.approx(.82,abs=1e-4,rel=0)


@pytest.mark.parametrize("p", [-2., -.1, 0.])
def test_associativity_commutativity_and_duality(p):
    assert tn(.3,.7,p) == pytest.approx(tn(.7,.3,p),abs=1e-4,rel=0)
    assert tn(tn(.2,.5,p),.8,p) == pytest.approx(tn(.2,tn(.5,.8,p),p),abs=1e-4,rel=0)
    assert tc(.3,.7,p) == pytest.approx(1-tn(.7,.3,p),abs=1e-4,rel=0)


def test_positive_sign_conflict_is_explicit():
    with pytest.raises(ValueError,match="Gamma < 0"):
        tn(.2,.3,.5)
    with pytest.raises(ValueError,match="non-positive base"):
        tn(.1,.1,.5,positive="literal")
    assert tn(.1,.1,.5,positive="truncate") == 0


@pytest.mark.parametrize("a,b,p", [(-.1,.5,-1), (.5,1.1,-1), (float("nan"),.5,-1), (.2,.3,float("inf"))])
def test_bad_domain(a,b,p):
    with pytest.raises(ValueError):
        tn(a,b,p)


def test_extreme_negative_parameter_does_not_overflow():
    assert tn(1e-100,.8,-100) == pytest.approx(1e-100,rel=1e-10,abs=0)
