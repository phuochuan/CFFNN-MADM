"""Manually transcribed from the local PDF, visually checked at pp.8--10,12.

Axes: expert table, HIC alternative, LIS criterion, four CFF components.
Expected observations are validation fixtures, never computational inputs.
"""

from src.cff import CFFNumber

ALTERNATIVES = ("HIC1", "HIC2", "HIC3", "HIC4")
CRITERIA = ("LIS1", "LIS2", "LIS3", "LIS4")
CRITERION_WEIGHTS = (0.23, 0.24, 0.25, 0.28)

# Table 1, p.8. HIC1/LIS2 really prints 0.1 exp(i2pi*0.7).
TABLE1_RAW = (
    ((.1,.1,.3,.3), (.05,.05,.1,.7), (.04,.04,.8,.8), (.02,.02,.04,.04)),
    ((.05,.05,.7,.7), (.1,.1,.3,.3), (.02,.02,.04,.04), (.04,.04,.8,.8)),
    ((.04,.04,.8,.8), (.02,.02,.04,.04), (.05,.05,.7,.7), (.1,.1,.3,.3)),
    ((.02,.02,.04,.04), (.04,.04,.8,.8), (.1,.1,.3,.3), (.05,.05,.7,.7)),
)

# Table 2, p.8. Keep 0.024/0.028 distinct from 0.0024/0.0028.
TABLE2_RAW = (
    ((.02,.02,.03,.03), (.02,.02,.03,.03), (.02,.02,.03,.03), (.3,.3,.5,.5)),
    ((.02,.02,.03,.03), (.02,.02,.03,.03), (.3,.3,.5,.5), (.04,.04,.8,.8)),
    ((.024,.024,.028,.028), (.3,.3,.5,.5), (.05,.05,.7,.7), (.2,.2,.3,.3)),
    ((.3,.3,.5,.5), (.0024,.0024,.0028,.0028), (.1,.1,.3,.3), (.02,.02,.7,.7)),
)
DECISION_MATRICES = tuple(tuple(tuple(CFFNumber(*cell) for cell in row)
                                for row in table) for table in (TABLE1_RAW, TABLE2_RAW))

# Table 3, pp.8--9. Raw tuples intentionally allow auditing invalid observations.
TABLE3_EXPECTED = (
    ((.1091,.0245,.3881,.0212), (.1523,.1698,.1995,.0298),
     (.1916,.0217,.2361,.0126), (.0341,.1234,.0675,.5431)),
    ((.3092,.0456,.5066,.0269), (.0031,.0172,.8162,.0662),
     (.6891,.0963,.7784,.1297), (.3696,.4596,.5968,.0369)),
    ((.0198,.0198,.4911,.0563), (.8031,.0278,.9142,.0212),
     (.5567,.1296,.6364,.0249), (.2363,.0236,.9678,.0219)),
    ((.2991,.0297,.9913,.1298), (.7402,.7402,.9741,.9741),
     (.4125,.5894,.7741,.8741), (.8123,.8796,.8254,.9871)),
)

# Table 4, p.10 labels these rows LIS1--LIS4, not HIC1--HIC4.
# Store only; never feed this table into the pipeline to bypass Table 3.
TABLE4_EXPECTED = (
    (.7844,.1789,.3781,.4115), (.5693,.0459,.1785,.7841),
    (.2245,.2348,.2001,.1987), (.2191,.4471,.7819,.0745),
)
FINAL_EXPECTED = (.0199, .5478, .2979, .5148)
RANKING_EXPECTED = ("HIC2", "HIC4", "HIC3", "HIC1")
SENSITIVITY_EXPECTED = {
    .1: (.3814,.8155,.5991,.7719),
    .2: (.0319,.9015,.6371,.7998),
    .3: (.3008,.8201,.5103,.6908),
    .4: (.0004,.6015,.4503,.5698),
    .5: (.0023,.7896,.5509,.6872),
    .6: (.1117,.4905,.3508,.4627),
}
