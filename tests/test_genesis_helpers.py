import pytest

from genesis3d.genesis_sort_advanced import clamp_force


def test_force_scaling_respects_negative_limit():
    assert clamp_force(-1.5, 1.2, -3.0) == pytest.approx(-1.8)
    assert clamp_force(-1.5, 3.0, -3.0) == pytest.approx(-3.0)
