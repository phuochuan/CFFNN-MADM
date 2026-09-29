"""Four independent real components; phases are fractions of one full turn."""

from dataclasses import dataclass, astuple
from math import isfinite


@dataclass(frozen=True)
class CFFNumber:
    """Section 2, p.3: componentwise interpretation of the complex constraints.

    The printed order relation on complex numbers is undefined. We explicitly
    require mu^3 + nu^3 <= 1 separately for magnitudes and normalized phases.
    Invalid paper observations remain raw tuples in data/paper_case.py.
    """

    mu_magnitude: float
    mu_phase: float
    nu_magnitude: float
    nu_phase: float

    def __post_init__(self):
        if any(not isfinite(v) or not 0 <= v <= 1 for v in self.components):
            raise ValueError("CFF components must be finite numbers in [0, 1]")
        for label, mu, nu in (
            ("magnitude", self.mu_magnitude, self.nu_magnitude),
            ("phase", self.mu_phase, self.nu_phase),
        ):
            if mu**3 + nu**3 > 1 + 1e-12:
                raise ValueError(f"Fermatean {label} constraint violated: {mu**3 + nu**3:.12g} > 1")

    @property
    def components(self):
        return astuple(self)
