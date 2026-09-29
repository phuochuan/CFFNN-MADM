"""Second Definition 2.3, p.3: the printed score is complex, not ordered."""

from cmath import exp
from math import pi


def score(x):
    """D_m^3 exp(i2pi D_phase^3) - E_m^3 exp(i2pi E_phase^3).

    Splits overloaded D/E notation into the four independent components.
    Returns complex; the article provides no real-valued projection for ranking.
    Do not silently use real part, modulus, or a different scalar score.
    """
    return (x.mu_magnitude**3 * exp(2j*pi*x.mu_phase**3)
            - x.nu_magnitude**3 * exp(2j*pi*x.nu_phase**3))
