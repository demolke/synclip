"""Tests for the per-modifier influence-mode UI in the modifier stack."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from synclip.modifiers import ModifierConfig  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def _stack(*configs):
    from synclip.ui.modifier_stack import ModifierStackWidget
    w = ModifierStackWidget()
    w.bind(SimpleNamespace(modifiers=list(configs)))
    return w


def _set_anim(row, mode: str) -> None:
    i = row._anim_combo.findData(mode)
    row._anim_combo.setCurrentIndex(i)
    row._on_anim_changed(i)


def test_row_defaults_to_static_influence(qapp):
    w = _stack(ModifierConfig("smooth", influence=0.5))
    row = w._rows[0]
    assert row._anim_combo.currentData() == "off"
    assert not row._infl_slider.isHidden()
    assert row._readout.isHidden()
    assert row._curve_btn.isHidden()


def test_switch_to_absolute_swaps_to_readout_and_reseeds_curve(qapp):
    mc = ModifierConfig("smooth", influence=0.4)
    w = _stack(mc)
    row = w._rows[0]
    _set_anim(row, "absolute")
    assert mc.influence_anim == "absolute"
    assert mc.influence_curve == [[0.0, 0.4], [1.0, 0.4]]   # flat at base
    assert row._infl_slider.isHidden() and not row._readout.isHidden()
    assert not row._curve_btn.isHidden()


def test_switch_to_relative_reseeds_flat_zero_offset(qapp):
    mc = ModifierConfig("smooth", influence=0.4, influence_anim="absolute",
                        influence_curve=[[0.0, 0.4], [1.0, 0.9]])
    w = _stack(mc)
    row = w._rows[0]
    _set_anim(row, "relative")
    assert mc.influence_curve == [[0.0, 0.0], [1.0, 0.0]]   # neutral offset
    assert not row._infl_slider.isHidden() and row._readout.isHidden()


def test_curve_button_selects_and_deselects(qapp):
    mc = ModifierConfig("smooth", influence=0.5, influence_anim="relative")
    w = _stack(mc)
    seen = []
    w.influence_curve_selected.connect(seen.append)
    row = w._rows[0]
    row._curve_btn.setChecked(True)
    assert seen[-1] is mc
    row._curve_btn.setChecked(False)
    assert seen[-1] is None


def test_only_one_curve_active_at_a_time(qapp):
    a = ModifierConfig("smooth", influence=0.5, influence_anim="relative")
    b = ModifierConfig("input", influence=0.5, influence_anim="relative")
    w = _stack(a, b)
    r0, r1 = w._rows
    r0._curve_btn.setChecked(True)
    r1._curve_btn.setChecked(True)
    assert r1._curve_btn.isChecked() and not r0._curve_btn.isChecked()
    assert w._curve_row is r1


def test_set_row_influence_updates_readout(qapp):
    w = _stack(ModifierConfig("smooth", influence=0.5, influence_anim="absolute"))
    w.set_row_influence(0, 0.73)
    assert w._rows[0]._readout.text() == "0.73"


def test_rebuild_closes_active_editor(qapp):
    mc = ModifierConfig("smooth", influence=0.5, influence_anim="relative")
    w = _stack(mc)
    seen = []
    w.influence_curve_selected.connect(seen.append)
    w._rows[0]._curve_btn.setChecked(True)
    assert seen[-1] is mc
    w.bind(SimpleNamespace(modifiers=[mc]))   # rebuild
    assert seen[-1] is None and w._curve_row is None
