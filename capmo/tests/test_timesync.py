"""Clock-sync math + estimator tests."""

from __future__ import annotations

from ..timesync import compute_offset_rtt, ClockEstimator


def test_offset_and_rtt_with_known_values():
    # Phone clock leads the server by +100 ms; symmetric 20 ms each way.
    # server t0=1000; phone gets it at t1 = (1000+20)+100 = 1120; replies after
    # 5 ms phone time at t2=1125; server receives at t3 = (1125-100)+20 = 1045.
    offset, rtt = compute_offset_rtt(1000, 1120, 1125, 1045)
    assert abs(offset - 100.0) < 1e-9      # phone - server
    assert abs(rtt - 40.0) < 1e-9          # 45 wall - 5 think


def test_estimator_keeps_lowest_rtt_offset():
    est = ClockEstimator(window=8)
    est.add_exchange(0, 250, 260, 100)     # high rtt, noisy offset
    est.add_exchange(1000, 1120, 1125, 1045)  # rtt 40, offset 100
    assert est.ready
    assert abs(est.offset_ms - 100.0) < 1e-9
    assert abs(est.rtt_ms - 40.0) < 1e-9


def test_to_server_ms_uses_offset():
    est = ClockEstimator()
    est.add_exchange(1000, 1120, 1125, 1045)   # offset +100
    # A phone timestamp of 5000 is 4900 on the server timeline.
    assert abs(est.to_server_ms(5000) - 4900.0) < 1e-9
    assert abs(est.to_phone_ms(4900) - 5000.0) < 1e-9


def test_unready_estimator_is_zero_offset():
    est = ClockEstimator()
    assert not est.ready
    assert est.offset_ms == 0.0
    assert est.to_server_ms(123) == 123
