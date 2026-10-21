"""US EPA AQI conversion and the Punjab EPD report reader."""

from pathlib import Path

import numpy as np
import pytest

from saafhawa.validate_local import lahore_aqi, pm25_to_aqi

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("pm,aqi", [(0, 0), (9.0, 50), (9.1, 51), (35.4, 100), (35.5, 101), (55.4, 150),
                                    (55.5, 151), (125.4, 200), (125.5, 201), (225.4, 300), (225.5, 301), (325.4, 500), (900, 500)])
def test_epa_2024_breakpoints(pm, aqi):
    assert pm25_to_aqi(pm) == aqi


def test_nan_is_nan():
    assert np.isnan(pm25_to_aqi(float("nan")))


def test_reads_lahore_from_real_epd_report():
    """Runs only if `make data`/`validate` has downloaded at least one real EPD report."""
    reports = sorted((ROOT / "data" / "raw" / "punjab_epd").glob("*.pdf"))
    if not reports:
        pytest.skip("no EPD report downloaded yet (run: python -m saafhawa validate)")
    v = lahore_aqi(reports[-1])
    assert v is not None and 0 < v <= 999
