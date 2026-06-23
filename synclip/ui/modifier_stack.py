"""
ModifierStackWidget - the per-view modifier list shown in the right dock.

A source-stream selector plus an ordered list of modifier rows.  Each row has a
mute checkbox, an influence slider, reorder (up/down) and remove buttons, and an
expandable editor whose controls are *built automatically* from the modifier's
:class:`modifiers.ParamSpec` list.  A "+ Add modifier" button appends a new
modifier of any registered type.

The widget mutates the bound :class:`view_pipeline.ViewConfig` in place and emits
``changed`` so the host can re-apply the pipeline and refresh the previews.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .. import modifiers as mod_mod
from ..modifiers import ModifierConfig
from .curve_editor import CurveEditor


class _ParamEditor(QWidget):
    """Builds the editing control for one ParamSpec and writes back to params."""

    changed = Signal()

    def __init__(self, spec, params: dict, streams: list[str], parent=None) -> None:
        super().__init__(parent)
        self._spec = spec
        self._params = params
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        value = spec.coerce(params.get(spec.name))

        if spec.kind == "float":
            lay.addWidget(QLabel(spec.label or spec.name))
            self._label = QLabel(f"{value:.2f}")
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(int((value - spec.min) / (spec.max - spec.min) * 100))
            slider.valueChanged.connect(self._on_float)
            self._slider = slider
            lay.addWidget(slider, stretch=1)
            lay.addWidget(self._label)

        elif spec.kind == "bool":
            cb = QCheckBox(spec.label or spec.name)
            cb.setChecked(bool(value))
            cb.toggled.connect(self._on_bool)
            lay.addWidget(cb)

        elif spec.kind == "enum":
            lay.addWidget(QLabel(spec.label or spec.name))
            combo = QComboBox()
            for val, label in (spec.choices or []):
                combo.addItem(label, val)
            i = combo.findData(value)
            if i >= 0:
                combo.setCurrentIndex(i)
            combo.activated.connect(self._on_enum)
            self._combo = combo
            lay.addWidget(combo, stretch=1)

        elif spec.kind == "curve":
            box = QVBoxLayout()
            box.setContentsMargins(0, 0, 0, 0)
            box.addWidget(QLabel(spec.label or spec.name))
            curve = CurveEditor()
            curve.set_points([(p[0], p[1]) for p in value])
            curve.curve_changed.connect(self._on_curve)
            self._curve = curve
            box.addWidget(curve)
            lay.addLayout(box, stretch=1)

        elif spec.kind == "vec3bool":
            lay.addWidget(QLabel(spec.label or spec.name))
            self._checks = []
            for i, axis in enumerate("XYZ"):
                cb = QCheckBox(axis)
                cb.setChecked(bool(value[i]))
                cb.toggled.connect(self._on_vec3)
                self._checks.append(cb)
                lay.addWidget(cb)

        elif spec.kind == "streamset":
            lay.addWidget(QLabel(spec.label or spec.name))
            self._stream_checks = {}
            for name in streams:
                cb = QCheckBox(name)
                cb.setChecked(name in value)
                cb.toggled.connect(self._on_streamset)
                self._stream_checks[name] = cb
                lay.addWidget(cb)

    # -- write-back handlers -----------------------------------------------

    def _on_float(self, v: int) -> None:
        s = self._spec
        val = s.min + (v / 100.0) * (s.max - s.min)
        self._params[s.name] = val
        self._label.setText(f"{val:.2f}")
        self.changed.emit()

    def _on_bool(self, checked: bool) -> None:
        self._params[self._spec.name] = bool(checked)
        self.changed.emit()

    def _on_enum(self, _i: int) -> None:
        self._params[self._spec.name] = self._combo.currentData()
        self.changed.emit()

    def _on_curve(self, _lut) -> None:
        self._params[self._spec.name] = self._curve.points()
        self.changed.emit()

    def _on_vec3(self, _checked: bool) -> None:
        self._params[self._spec.name] = [cb.isChecked() for cb in self._checks]
        self.changed.emit()

    def _on_streamset(self, _checked: bool) -> None:
        self._params[self._spec.name] = [
            n for n, cb in self._stream_checks.items() if cb.isChecked()
        ]
        self.changed.emit()


class _ModifierRow(QFrame):
    """One modifier: header (mute/influence/reorder/edit/remove) + editor."""

    changed = Signal()
    move_up = Signal(object)
    move_down = Signal(object)
    remove = Signal(object)

    def __init__(self, mc: ModifierConfig, streams: list[str], parent=None) -> None:
        super().__init__(parent)
        self.mc = mc
        self._streams = streams
        self.setFrameShape(QFrame.Shape.Box)
        self.setLineWidth(1)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 4, 6, 4)
        outer.setSpacing(4)

        cls = mod_mod.get_class(mc.type)
        name = cls.display_name if cls else mc.type
        has_infl = cls.has_influence if cls else True

        header = QHBoxLayout()
        self._mute = QCheckBox()
        self._mute.setChecked(mc.enabled)
        self._mute.setToolTip("Enable / mute")
        self._mute.toggled.connect(self._on_mute)
        header.addWidget(self._mute)
        header.addWidget(QLabel(f"<b>{name}</b>"))

        if has_infl:
            self._infl_label = QLabel(f"{mc.influence:.2f}")
            infl = QSlider(Qt.Orientation.Horizontal)
            infl.setRange(0, 100)
            infl.setValue(int(mc.influence * 100))
            infl.setToolTip("Influence")
            infl.valueChanged.connect(self._on_influence)
            header.addWidget(infl, stretch=1)
            header.addWidget(self._infl_label)
        else:
            header.addStretch(1)

        self._edit_btn = QPushButton("E")
        self._edit_btn.setCheckable(True)
        self._edit_btn.setChecked(True)
        self._edit_btn.setFixedWidth(26)
        self._edit_btn.toggled.connect(self._on_edit_toggled)
        header.addWidget(self._edit_btn)
        up = QPushButton("^"); up.setFixedWidth(26)
        up.clicked.connect(lambda: self.move_up.emit(self))
        header.addWidget(up)
        down = QPushButton("v"); down.setFixedWidth(26)
        down.clicked.connect(lambda: self.move_down.emit(self))
        header.addWidget(down)
        rm = QPushButton("X"); rm.setFixedWidth(26)
        rm.clicked.connect(lambda: self.remove.emit(self))
        header.addWidget(rm)
        outer.addLayout(header)

        # Collapsible editor body built from the modifier's param specs.
        self._body = QWidget(self)
        body_lay = QVBoxLayout(self._body)
        body_lay.setContentsMargins(18, 0, 0, 2)
        body_lay.setSpacing(2)
        self._status = QLabel("")
        self._status.setStyleSheet("color:#888;")
        specs = cls.param_specs if cls else []
        for spec in specs:
            ed = _ParamEditor(spec, mc.params, streams, self._body)
            ed.changed.connect(self._on_param_changed)
            body_lay.addWidget(ed)
        body_lay.addWidget(self._status)
        self._body.setVisible(True)
        outer.addWidget(self._body)
        self._refresh_muted_style()

    def set_status(self, text: str) -> None:
        self._status.setText(text)

    def _on_mute(self, checked: bool) -> None:
        self.mc.enabled = checked
        self._refresh_muted_style()
        self.changed.emit()

    def _on_influence(self, v: int) -> None:
        self.mc.influence = v / 100.0
        self._infl_label.setText(f"{self.mc.influence:.2f}")
        self.changed.emit()

    def _on_edit_toggled(self, checked: bool) -> None:
        self._body.setVisible(checked)

    def _on_param_changed(self) -> None:
        self.changed.emit()

    def _refresh_muted_style(self) -> None:
        self.setStyleSheet("" if self.mc.enabled else "color:#666;")


class ModifierStackWidget(QWidget):
    """Source selector + the ordered modifier rows for one view."""

    changed = Signal()   # config mutated; host re-applies pipeline + refreshes

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._cfg = None
        self._streams = ["mediapipe", "ai", "retarget"]
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(2, 2, 2, 2)
        self._lay.setSpacing(3)

        self._rows_container = QWidget(self)
        self._rows_lay = QVBoxLayout(self._rows_container)
        self._rows_lay.setContentsMargins(0, 0, 0, 0)
        self._rows_lay.setSpacing(6)
        self._lay.addWidget(self._rows_container)

        self._add_btn = QPushButton("+ Add modifier")
        self._add_btn.clicked.connect(self._on_add_clicked)
        self._lay.addWidget(self._add_btn)
        self._lay.addStretch(1)
        self._rows: list[_ModifierRow] = []

    def set_available_streams(self, names: list[str]) -> None:
        self._streams = list(names)

    def bind(self, view_config) -> None:
        """Bind to a ViewConfig and rebuild the rows."""
        self._cfg = view_config
        self._rebuild_rows()

    def set_row_status(self, index: int, text: str) -> None:
        if 0 <= index < len(self._rows):
            self._rows[index].set_status(text)

    # -- internals ----------------------------------------------------------

    def _rebuild_rows(self) -> None:
        for row in self._rows:
            row.setParent(None)
            row.deleteLater()
        self._rows = []
        if self._cfg is None:
            return
        for mc in self._cfg.modifiers:
            row = _ModifierRow(mc, self._streams, self._rows_container)
            row.changed.connect(self.changed)
            row.move_up.connect(self._on_move_up)
            row.move_down.connect(self._on_move_down)
            row.remove.connect(self._on_remove)
            self._rows_lay.addWidget(row)
            self._rows.append(row)

    def _on_add_clicked(self) -> None:
        menu = QMenu(self)
        for type_name, label in mod_mod.available_types():
            if type_name.startswith("_"):
                continue
            act = menu.addAction(label)
            act.triggered.connect(lambda _=False, t=type_name: self._add_modifier(t))
        menu.exec(self._add_btn.mapToGlobal(self._add_btn.rect().bottomLeft()))

    def _add_modifier(self, type_name: str) -> None:
        if self._cfg is None:
            return
        cls = mod_mod.get_class(type_name)
        if cls is None:
            return
        self._cfg.modifiers.append(cls.default_config())
        self._rebuild_rows()
        self.changed.emit()

    def _on_move_up(self, row: _ModifierRow) -> None:
        self._move(row, -1)

    def _on_move_down(self, row: _ModifierRow) -> None:
        self._move(row, +1)

    def _move(self, row: _ModifierRow, delta: int) -> None:
        if self._cfg is None:
            return
        mods = self._cfg.modifiers
        i = mods.index(row.mc)
        j = i + delta
        if 0 <= j < len(mods):
            mods[i], mods[j] = mods[j], mods[i]
            self._rebuild_rows()
            self.changed.emit()

    def _on_remove(self, row: _ModifierRow) -> None:
        if self._cfg is None:
            return
        self._cfg.modifiers.remove(row.mc)
        self._rebuild_rows()
        self.changed.emit()
