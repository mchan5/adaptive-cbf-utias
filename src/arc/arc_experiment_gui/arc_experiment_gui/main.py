"""arc_experiment_gui -- single-screen operator console. One screen, no tabs."""

import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime

import rclpy
from ament_index_python.packages import get_package_share_directory
from PyQt5.QtCore import Qt, QElapsedTimer, QTimer
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from px4_msgs.msg import VehicleStatus

from arc_experiment_gui.campaign import Manifest, DEFAULT_ARMS, git_state
from arc_experiment_gui.launcher import StackLauncher, BagRecorder
from arc_experiment_gui.obstacle_view import ObstacleView
from arc_experiment_gui.ros_node import TelemetryNode, RosSpinThread
from arc_experiment_gui.single_run import SingleRunner

# Workspace root, used only as a run-from-source fallback when the installed
# package share dir isn't found. $ARC_ROOT is exported by scripts/setup.sh.
_ARC_ROOT = os.environ.get('ARC_ROOT', os.path.expanduser('~/arc_2026'))
_REPO = os.environ.get('ARC_CBF_REPO', os.path.join(_ARC_ROOT, 'src', 'arc'))

_ARMED = VehicleStatus.ARMING_STATE_ARMED
_OFFBOARD = VehicleStatus.NAVIGATION_STATE_OFFBOARD

_ALARM_M = 0.30
_WARN_M = 1.00

# START RUN sequence -- maps a runner phase to a step index in the stepper.
_SEQ = [
    ('set_params', 'push arm params to CBF node (penn / γ)'),
    ('set_scene', 'push scene obstacles (param, no relaunch)'),
    ('return_to_start', 'fly back to start point  (SITL only)'),
    ('record_start', 'start rosbag → <label>_arm_trialK/'),
    ('goto_goal', 'move setpoint to goal → vehicle flies'),
    ('wait_reached', 'wait for reached / timeout'),
    ('await_outcome', 'confirm outcome → append manifest.json'),
]
_PHASE_STEP = {
    'set_params': 0, 'set_scene': 1, 'return_to_start': 2,
    'record_start': 3, 'goto_goal': 4, 'wait_reached': 5, 'await_outcome': 6,
}


def _slug(text):
    """Filesystem/manifest-safe token from a freeform LiDAR arrangement label."""
    return re.sub(r'[^A-Za-z0-9._-]+', '-', text.strip()).strip('-') or 'unlabelled'


def _scene_dir():
    # ARC_SCENE_DIR overrides the packaged set -- point it at a frozen results dir (e.g.
    # .../hardware_scenes_final_20260831/scenes) to fly that scene set without re-copying files …
    env = os.environ.get('ARC_SCENE_DIR')
    if env:
        return os.path.expanduser(env)
    try:
        return os.path.join(
            get_package_share_directory('single_vehicle_cbf_rate_arc'),
            'config', 'scenes', 'hw_campaign')
    except Exception:  # noqa: BLE001
        return os.path.join(_REPO, 'single_vehicle_cbf_rate_arc',
                            'config', 'scenes', 'hw_campaign')


def _yaw_from_quat(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def _set_style(w, s):
    """setStyleSheet only when the string actually changes."""
    if w.styleSheet() != s:
        w.setStyleSheet(s)


def _chip(text, ok=None):
    lbl = QLabel(text)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setMargin(4)
    if ok is True:
        lbl.setStyleSheet('background:#1f5130;color:#dfffe6;border-radius:4px;font-weight:bold;font-size:12px;padding:2px 4px;')
    elif ok is False:
        lbl.setStyleSheet('background:#5a1f1f;color:#ffdede;border-radius:4px;font-weight:bold;font-size:12px;padding:2px 4px;')
    else:
        lbl.setStyleSheet('background:#2a2f36;color:#dfe3e8;border-radius:4px;font-size:12px;padding:2px 4px;')
    return lbl


class OutcomeDialog(QDialog):
    def __init__(self, info, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Trial outcome')
        row = info['row']
        v = QVBoxLayout(self)
        v.addWidget(QLabel(
            f"scene {row['scene_id']}  ·  arm {row['arm']}  ·  trial {row['trial_index']}\n"
            f"t_goal = {row['t_goal']} s   min_obs = {row['min_obstacle_dist']} m   "
            f"penn_duty = {row['penn_duty_cycle']}\n"
            f"suggested: {info['suggested']}"))
        self._combo = QComboBox()
        self._combo.addItems(['reached', 'dnf', 'collision', 'aborted'])
        self._combo.setCurrentText(info['suggested'] if info['suggested'] in
                                   ('reached', 'dnf') else 'reached')
        v.addWidget(self._combo)
        self._notes = QLineEdit()
        self._notes.setPlaceholderText('notes (optional)')
        v.addWidget(self._notes)
        bb = QDialogButtonBox(QDialogButtonBox.Ok)
        bb.accepted.connect(self.accept)
        v.addWidget(bb)

    def result_tuple(self):
        return self._combo.currentText(), self._notes.text()


# Companion-stack nodes to show a health dot for, by regime. Union is built once; _slow_tick only
# checks/shows the ones for the active regime.
_HEALTH_COMMON = (
    ('autopilot_sv_cbf_rate_node', 'cbf_rate_node'),
)
# The odom source and the arming driver differ by regime.
_HEALTH_HW_IFACE = (
    ('mocap_odom_bridge_node', 'odom_bridge'),
    ('operator_arm_node', 'arm_node'),
)
_HEALTH_SITL_IFACE = (
    ('px4_odom_bridge_node', 'odom_bridge'),
    ('ground_station_stub_node', 'gcs_stub'),
)
_HEALTH_BY_REGIME = {
    'synthetic': _HEALTH_COMMON + _HEALTH_HW_IFACE
    + (('synthetic_obstacle_publisher', 'obstacle_pub'),),
    'lidar': _HEALTH_COMMON + _HEALTH_HW_IFACE
    + (('lidar_obstacle_publisher', 'lidar_pub'),),   # launch remaps to this name
    'sitl': _HEALTH_COMMON + _HEALTH_SITL_IFACE
    + (('synthetic_obstacle_publisher', 'obstacle_pub'),),
}
_HEALTH_NODES = tuple(dict(
    _HEALTH_BY_REGIME['synthetic'] + _HEALTH_BY_REGIME['lidar']
    + _HEALTH_BY_REGIME['sitl']).items())


class Console(QWidget):

    def __init__(self, node):
        super().__init__()
        self._node = node
        self._launcher = StackLauncher()
        self._bag = BagRecorder()
        self._runner = SingleRunner(node, self._bag)

        self._armed = None
        self._last_feasible = None
        self._last_reached = None
        self._last_min_dist = None
        self._params_synced = False
        self._param_poll = QElapsedTimer(); self._param_poll.start()
        self._odom_timer = QElapsedTimer()
        self._have_odom = False
        self._t_diag = self._t_odom = self._t_obs = 0.0
        self._regime = 'synthetic'
        self._session_dir = ''

        root = QVBoxLayout(self)
        root.addLayout(self._build_topbar())
        root.addWidget(self._bar)
        root.addWidget(self._alarm)
        body = QHBoxLayout()
        body.addWidget(self._build_left(), 0)
        body.addWidget(self._build_centre(), 1)
        body.addWidget(self._build_right(), 0)
        root.addLayout(body, 1)

        # wire ROS signals
        node.diag.connect(self._on_diag)
        node.odom.connect(self._on_odom)
        node.obstacles.connect(self._on_obstacles)
        node.status.connect(self._on_status)
        node.battery.connect(self._on_battery)
        node.params_read.connect(self._on_params_read)
        node.params_result.connect(self._on_params_result)
        node.diag.connect(lambda _m: setattr(self, '_t_diag', time.time()))
        node.odom.connect(lambda _m: setattr(self, '_t_odom', time.time()))
        node.obstacles.connect(lambda _m: setattr(self, '_t_obs', time.time()))

        self._launcher.line.connect(self._log.appendPlainText)
        self._launcher.state.connect(self._on_stack_state)
        self._bag.line.connect(self._log.appendPlainText)

        self._runner.state.connect(self._on_run_state)
        self._runner.progress.connect(self._log_line)
        self._runner.awaiting_outcome.connect(self._on_await_outcome)
        self._runner.trial_written.connect(self._on_trial_written)
        self._runner.run_done.connect(self._on_run_done)

        self._refresh_config_label()
        self._sync_regime()
        self._sync_arm(quiet=True)

        self._fast = QTimer(self); self._fast.timeout.connect(self._on_ui_tick)
        self._fast.start(200)
        self._slow = QTimer(self); self._slow.timeout.connect(self._slow_tick)
        self._slow.start(500)
        self._log_line('console started -- waiting for telemetry')

    # top bar
    def _build_topbar(self):
        # Two stacked rows so the strip never forces a wide minimum window: row 1 = controls, row 2
        # = status chips.
        outer = QVBoxLayout()
        outer.setSpacing(4)

        row1 = QHBoxLayout()
        brand = QLabel('ARC Operator Console')
        brand.setStyleSheet('font-weight:bold;font-size:14px;')
        row1.addWidget(brand)
        row1.addWidget(QLabel(f'{self._node.uav_prefix} · adaptive-γ'))
        row1.addSpacing(10)
        row1.addWidget(QLabel('regime'))
        self._obs = QComboBox()
        self._obs.addItems([
            'synthetic (known positions)', 'LiDAR — live', 'SITL (gz_x500)'])
        self._obs.currentTextChanged.connect(self._sync_regime)
        row1.addWidget(self._obs)
        row1.addWidget(QLabel('barrier'))
        self._barrier = QComboBox()
        self._barrier.addItems(['sphere', 'cylinder'])
        self._barrier.setToolTip(
            'CBF obstacle model (SITL and hardware). sphere = original 3-D '
            'barrier; cylinder = vertical-cylinder (horizontal-distance) '
            'barrier -- only safe with genuinely floor-to-ceiling obstacles, '
            'and NOT yet flown, so SITL-rehearse first. Passed as '
            'cbf_cylinder_barrier:= to the launch at connect; it is '
            'structural, so changing it needs a reconnect.')
        row1.addWidget(self._barrier)
        self._btn_conn = QPushButton('connect')
        self._btn_conn.clicked.connect(self._on_connect)
        row1.addWidget(self._btn_conn)
        self._conn_pill = _chip('not connected', None)
        row1.addWidget(self._conn_pill)
        row1.addStretch(1)

        row2 = QHBoxLayout()
        self._c_arm = _chip('ARM: ?')
        self._c_offb = _chip('OFFBOARD: ?')
        self._c_feas = _chip('QP: ?')
        self._c_mode = _chip('mode: ?')
        self._c_gamma = _chip('γ: ?')
        self._c_goal = _chip('goal: ?')
        self._c_ref = _chip('SITL ref: ?')
        for c in (self._c_arm, self._c_offb, self._c_feas, self._c_mode,
                  self._c_gamma, self._c_goal, self._c_ref):
            row2.addWidget(c)
        self._lbl_cfg = QLabel('config: ?')
        self._lbl_cfg.setStyleSheet('color:#9aa0a6;')
        row2.addWidget(self._lbl_cfg)
        row2.addStretch(1)

        outer.addLayout(row1)
        outer.addLayout(row2)

        # safety strip
        self._bar = QProgressBar()
        self._bar.setRange(0, 300)
        self._bar.setFormat('min obstacle dist: %v cm')
        self._alarm = QLabel('')
        self._alarm.setAlignment(Qt.AlignCenter)
        self._alarm.setStyleSheet('color:#ff6b6b;font-weight:bold;')
        return outer

    # left column
    def _build_left(self):
        col = QWidget(); col.setFixedWidth(330)
        v = QVBoxLayout(col); v.setContentsMargins(0, 0, 0, 0)

        run = QGroupBox('run a trial  ·  pick · press · fly')
        form = QFormLayout()
        self._scene = QComboBox()
        for p in sorted(glob.glob(os.path.join(_scene_dir(), 'scene_*.json'))):
            if p.endswith('.ref.json'):
                continue
            self._scene.addItem(os.path.basename(p), p)
        self._scene.currentIndexChanged.connect(self._on_scene_changed)
        form.addRow('scene', self._scene)
        self._lidar_lbl = QLabel('arrangement')
        self._lidar_label = QLineEdit()
        self._lidar_label.setPlaceholderText('freeform label, e.g. wall-gap-A  (lidar only)')
        form.addRow(self._lidar_lbl, self._lidar_label)
        self._arm = QComboBox(); self._arm.addItems(list(DEFAULT_ARMS))
        self._arm.currentTextChanged.connect(lambda _t: self._sync_arm())
        form.addRow('arm', self._arm)
        self._arm_preview = QLabel('→ ?')
        self._arm_preview.setStyleSheet('color:#9aa0a6;font-family:monospace;')
        form.addRow(self._arm_preview)
        self._regime_note = QLabel('regime ?')
        self._regime_note.setWordWrap(True)
        self._regime_note.setStyleSheet('color:#9aa0a6;font-family:monospace;')
        form.addRow(self._regime_note)

        self._adv = QGroupBox('advanced — ad-hoc γ (tags runs arm=adhoc)')
        self._adv.setCheckable(True)
        self._adv.setChecked(False)
        av = QHBoxLayout()
        self._spin_gamma = QDoubleSpinBox()
        self._spin_gamma.setRange(0.5, 12.0)
        self._spin_gamma.setSingleStep(0.5)
        self._spin_gamma.setDecimals(2)
        self._spin_gamma.setValue(6.5)
        self._btn_gamma = QPushButton('apply live')
        self._btn_gamma.clicked.connect(self._on_apply_gamma)
        av.addWidget(QLabel('try γ')); av.addWidget(self._spin_gamma)
        av.addWidget(self._btn_gamma)
        self._adv.setLayout(av)
        form.addRow(self._adv)

        self._chk_record = QCheckBox('record rosbag for this run')
        self._chk_record.setChecked(True)
        form.addRow(self._chk_record)

        self._btn_run = QPushButton('●  START RUN')
        self._btn_run.setStyleSheet(
            'font-weight:bold;font-size:14px;padding:10px;'
            'background:#1c6b84;color:#eafcff;border-radius:6px;')
        self._btn_run.clicked.connect(self._on_run)

        self._seq_lbls = []
        seqbox = QVBoxLayout()
        for _ph, text in _SEQ:
            lbl = QLabel('·  ' + text)
            lbl.setStyleSheet('color:#8a8f98;font-family:monospace;font-size:11px;')
            self._seq_lbls.append(lbl)
            seqbox.addWidget(lbl)

        note = QLabel('One press does the whole sequence. Each run appends a row '
                      'to manifest.json and writes its bag under the session dir. '
                      'STOP cancels the run, closes the bag; vehicle holds.')
        note.setWordWrap(True)
        note.setStyleSheet('color:#9aa0a6;font-size:11px;')

        rv = QVBoxLayout()
        rv.addLayout(form)
        rv.addWidget(self._btn_run)
        rv.addLayout(seqbox)
        rv.addWidget(note)
        run.setLayout(rv)
        v.addWidget(run)

        pre = QGroupBox('pre-flight')
        pv = QVBoxLayout()
        self._checks = [
            ('CBF node up', lambda: 'autopilot_sv_cbf_rate_node' in self._node.graph_node_names()),
            ('odom fresh', lambda: time.time() - self._t_odom < 2.0),
            ('diagnostics streaming', lambda: time.time() - self._t_diag < 2.0),
            ('obstacles publishing', lambda: time.time() - self._t_obs < 3.0),
            ('exactly 1 obstacle publisher', lambda: self._node.obstacle_publisher_count() == 1),
            ('param service ready', self._node.params_service_ready),
            ('scene push service ready', self._node.scene_service_ready),
            ('disk > 2 GB free', lambda: shutil.disk_usage(os.path.expanduser('~')).free > 2e9),
        ]
        self._check_lbls = []
        for name, _fn in self._checks:
            lbl = QLabel('·  ' + name)
            lbl.setStyleSheet('font-family:monospace;font-size:11px;color:#8a8f98;')
            self._check_lbls.append(lbl)
            pv.addWidget(lbl)
        self._ready = QLabel('NOT READY')
        self._ready.setStyleSheet('font-weight:bold;color:#d08a5a;')
        pv.addWidget(self._ready)
        self._health = {}
        hrow = QHBoxLayout()
        for nid, short in _HEALTH_NODES:
            lbl = QLabel('○ ' + short)
            lbl.setStyleSheet('font-family:monospace;font-size:11px;color:#8a8f98;')
            self._health[nid] = lbl
            hrow.addWidget(lbl)
        pv.addLayout(hrow)
        pre.setLayout(pv)
        v.addWidget(pre)
        v.addStretch(1)
        return col

    # centre column
    def _build_centre(self):
        col = QWidget()
        v = QVBoxLayout(col); v.setContentsMargins(0, 0, 0, 0)
        vb = QGroupBox('corridor · top-down (ENU)')
        vl = QVBoxLayout()
        self._view = ObstacleView()
        vl.addWidget(self._view)
        vb.setLayout(vl)
        v.addWidget(vb, 1)

        st = QGroupBox('vehicle state  ·  cbf/diagnostics + odom')
        grid = QGridLayout()
        self._v = {k: QLabel('--') for k in (
            'pos', 'vel', 'thrust', 'delta', 'dhat', 'dist_goal', 'odom_age', 'batt')}
        labels = [('pos [m]', 'pos'), ('|vel| [m/s]', 'vel'), ('thrust [N]', 'thrust'),
                  ('QP slack', 'delta'), ('d_hat [N]', 'dhat'), ('dist goal [m]', 'dist_goal'),
                  ('odom age [s]', 'odom_age'), ('battery', 'batt')]
        for i, (text, key) in enumerate(labels):
            grid.addWidget(QLabel(text), i // 4, (i % 4) * 2)
            grid.addWidget(self._v[key], i // 4, (i % 4) * 2 + 1)
        st.setLayout(grid)
        v.addWidget(st)
        return col

    # right column
    def _build_right(self):
        col = QWidget(); col.setFixedWidth(350)
        v = QVBoxLayout(col); v.setContentsMargins(0, 0, 0, 0)

        cov = QGroupBox('coverage  ·  1 row / START RUN, from manifest.json')
        cv = QVBoxLayout()
        self._table = QTableWidget()
        self._table.setColumnCount(len(DEFAULT_ARMS))
        self._table.setHorizontalHeaderLabels(list(DEFAULT_ARMS))
        cv.addWidget(self._table)
        brow = QHBoxLayout()
        self._cov_lbl = QLabel('0 rows')
        self._cov_lbl.setStyleSheet('color:#9aa0a6;font-family:monospace;font-size:11px;')
        brow.addWidget(self._cov_lbl, 1)
        self._btn_zip = QPushButton('zip session dir')
        self._btn_zip.clicked.connect(self._on_zip)
        brow.addWidget(self._btn_zip)
        cv.addLayout(brow)
        cov.setLayout(cv)
        v.addWidget(cov)

        logbox = QGroupBox('event log')
        lv = QVBoxLayout()
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(4000)
        self._log.setStyleSheet('font-family:monospace;font-size:11px;')
        lv.addWidget(self._log)
        logbox.setLayout(lv)
        v.addWidget(logbox, 1)
        return col

    # helpers
    def _log_line(self, text):
        ts = self._node.get_clock().now().nanoseconds * 1e-9
        self._log.appendPlainText(f'[{ts:12.2f}] {text}')

    def _refresh_config_label(self):
        try:
            h = subprocess.run(['git', '-C', _REPO, 'rev-parse', '--short', 'HEAD'],
                               capture_output=True, text=True, timeout=3).stdout.strip()
            dirty = subprocess.run(['git', '-C', _REPO, 'status', '--porcelain'],
                                   capture_output=True, text=True, timeout=3).stdout
            n = len([ln for ln in dirty.splitlines() if ln.strip()])
        except Exception:  # noqa: BLE001
            self._lbl_cfg.setText('config: (git unavailable)')
            return
        if not h:
            self._lbl_cfg.setText('config: (not a git repo)')
        elif n:
            self._lbl_cfg.setText(f'config: {h}  ⚠ DIRTY ({n})')
            self._lbl_cfg.setStyleSheet('color:#ffb454;font-weight:bold;')
        else:
            self._lbl_cfg.setText(f'config: {h}  clean')

    def _sync_regime(self, *_):
        text = self._obs.currentText()
        if text.startswith('LiDAR'):
            regime = 'lidar'
        elif text.startswith('SITL'):
            regime = 'sitl'
        else:
            regime = 'synthetic'
        self._regime = regime
        lidar = regime == 'lidar'
        self._barrier.setEnabled(not self._launcher.running())
        self._scene.setEnabled(not lidar)
        self._lidar_lbl.setVisible(lidar)
        self._lidar_label.setVisible(lidar)
        self._session_dir = os.path.expanduser(
            f'~/arc_campaigns/{datetime.now():%Y%m%d}_{regime}')
        self._regime_note.setText(f'regime {regime}  ·  out {self._session_dir}')
        self._on_scene_changed()
        self._reload_matrix()

    def _on_scene_changed(self, *_):
        """Load the frozen scene's expected obstacles + (if exported) the SITL
        reference trajectory into the top-down view. Graceful if absent."""
        if self._regime == 'lidar':
            self._view.set_expected_obstacles([])
            self._view.set_reference_trail([])
            self._c_ref.setText('SITL ref: n/a (lidar)')
            return
        flat = self._load_scene(self._scene.currentData()) or []
        self._view.set_expected_obstacles(
            [(flat[i], flat[i + 1], flat[i + 3] / 2.0)
             for i in range(0, len(flat) - 3, 4)])
        ref = self._load_reference(self._scene.currentData())
        if ref:
            self._view.set_reference_trail(ref.get('trail', []))
            self._c_ref.setText(
                f"SITL ref: t_goal {ref.get('t_goal', '?')} s · "
                f"min {ref.get('min_obstacle_dist', '?')} m")
        else:
            self._view.set_reference_trail([])
            self._c_ref.setText('SITL ref: not exported')

    def _load_reference(self, scene_path):
        if not scene_path:
            return None
        ref_path = re.sub(r'\.json$', '.ref.json', scene_path)
        try:
            with open(ref_path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def _sync_arm(self, quiet=False):
        arm = self._arm.currentText()
        c = DEFAULT_ARMS.get(arm, {})
        if c.get('penn_enabled'):
            txt = 'penn_enabled=true · γ from PENN'
        else:
            txt = f'penn_enabled=false · cbf_gamma_obs={c.get("cbf_gamma_obs")}'
        self._arm_preview.setText('→ ' + txt)
        self._c_mode.setText('mode: adaptive (PENN)' if c.get('penn_enabled')
                             else f'mode: fixed γ {c.get("cbf_gamma_obs")}')
        if quiet:
            return
        if self._params_synced and self._node.params_service_ready():
            if not self._guard_ok(f'arm {arm}'):
                return
            self._log_line(f'-> arm {arm}: set {txt} (live, next tick)')
            self._node.set_params(dict(c))

    def _guard_ok(self, action):
        d = self._last_min_dist
        if self._armed and d is not None and d < _WARN_M:
            r = QMessageBox.question(
                self, 'Confirm mid-flight change',
                f'Vehicle is {d:.2f} m from an obstacle.\nApply "{action}" now?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            return r == QMessageBox.Yes
        return True

    def _load_scene(self, path):
        try:
            with open(path) as f:
                flat = list(json.load(f))
            if len(flat) % 4 != 0:
                raise ValueError(f'scene length {len(flat)} not a multiple of 4')
            return [float(x) for x in flat]
        except Exception as exc:  # noqa: BLE001
            self._log_line(f'scene load failed ({path}): {exc}')
            return None

    # session connect
    def _on_connect(self):
        target = 'sitl' if self._regime == 'sitl' else 'hardware'
        if self._launcher.running():
            self._log_line(f'session: tearing down {target} stack')
            self._launcher.stop()
            return
        osrc = 'lidar' if self._regime == 'lidar' else 'manual'
        scene_path = '' if self._regime == 'lidar' else (self._scene.currentData() or '')
        barrier = self._barrier.currentText()
        cylinder = barrier == 'cylinder'
        if target == 'sitl':
            self._log_line(
                f'session: bringing up SITL stack (obstacle_source={osrc}, '
                f'barrier={barrier}). '
                'PX4 SITL + gz_x500 + MicroXRCEAgent are started by the launch '
                'file; arm_mode=auto, GUI owns goal_waypoint.')
        else:
            self._log_line(
                f'session: bringing up hardware stack (obstacle_source={osrc}, '
                f'barrier={barrier}). '
                'MicroXRCEAgent / Livox driver / OptiTrack must already be up.'
                + ('  WARNING: cylinder barrier is not flight-validated -- '
                   'sphere is the certified model.' if cylinder else ''))
        self._launcher.start(scene_path, osrc, False, target=target,
                             cylinder_barrier=cylinder)

    def _on_stack_state(self, s):
        running = s in ('starting', 'running', 'stopping')
        self._btn_conn.setText('disconnect' if running else 'connect')
        self._obs.setEnabled(not running)
        self._barrier.setEnabled(not running)
        pill = {'starting': ('bringing up…', None), 'running': ('stack up', True),
                'stopping': ('stopping…', None), 'stopped': ('not connected', False)}
        txt, ok = pill.get(s, (s, None))
        self._conn_pill.setText(txt)
        self._conn_pill.setStyleSheet(_chip('', ok).styleSheet())
        self._log.appendPlainText(f'[stack: {s}]')

    # START RUN
    def _on_run(self):
        if self._runner.busy():
            self._runner.abort()
            return
        if not self._params_synced or not self._node.params_service_ready():
            QMessageBox.warning(self, 'not ready',
                                'CBF node parameters not in sync yet -- connect '
                                'the session and wait for telemetry.')
            return
        if not self._guard_ok('START RUN'):
            return
        adhoc = self._adv.isChecked()
        if adhoc:
            arm = 'adhoc'
            cfg = {'penn_enabled': False,
                   'cbf_gamma_obs': round(self._spin_gamma.value(), 3)}
        else:
            arm = self._arm.currentText()
            cfg = dict(DEFAULT_ARMS.get(arm, {}))

        if self._regime == 'lidar':
            label = self._lidar_label.text().strip()
            if not label:
                QMessageBox.warning(
                    self, 'label required',
                    'Enter an arrangement label for this LiDAR run '
                    '(it names the bag dir and the manifest scene_id).')
                return
            sid = None if adhoc else _slug(label)
            scene_flat = None
        else:
            sid = None if adhoc else self._scene.currentIndex()
            scene_flat = self._load_scene(self._scene.currentData())

        record = self._chk_record.isChecked()
        self._log_line(
            f'run start -- scene {sid} · arm {arm} · regime {self._regime}'
            + ('' if record else ' · NO BAG'))
        self._runner.start(self._session_dir, sid, arm, cfg, scene_flat,
                           self._regime, git_state(_REPO), record=record)

    def _on_run_state(self, phase):
        idx = _PHASE_STEP.get(phase, -1)
        for i, lbl in enumerate(self._seq_lbls):
            _ph, text = _SEQ[i]
            if i < idx:
                mark, col = '✓  ', '#8fce8f'
            elif i == idx:
                mark, col = '▸  ', '#eafcff'
            else:
                mark, col = '·  ', '#8a8f98'
            lbl.setText(mark + text)
            lbl.setStyleSheet(f'color:{col};font-family:monospace;font-size:11px;')
        running = phase not in ('idle', 'done')
        self._btn_run.setText('■  STOP RUN' if running else '●  START RUN')

    def _on_await_outcome(self, info):
        dlg = OutcomeDialog(info, self)
        dlg.exec_()
        outcome, notes = dlg.result_tuple()
        self._runner.submit_outcome(outcome, notes)

    def _on_trial_written(self, row):
        self._log_line(f'manifest row written: scene {row["scene_id"]} '
                       f'{row["arm"]} -> {row["outcome"]}  ({row["bag_path"]})')
        self._reload_matrix()

    def _on_run_done(self):
        for i, (_ph, text) in enumerate(_SEQ):
            self._seq_lbls[i].setText('·  ' + text)
            self._seq_lbls[i].setStyleSheet(
                'color:#8a8f98;font-family:monospace;font-size:11px;')
        self._btn_run.setText('●  START RUN')

    # coverage matrix
    def _reload_matrix(self):
        arms = list(DEFAULT_ARMS)
        rows = []
        mpath = os.path.join(self._session_dir, 'manifest.json')
        if os.path.exists(mpath):
            try:
                rows = Manifest(self._session_dir).rows
            except Exception:  # noqa: BLE001
                rows = []
        by_key = {}
        for r in rows:
            by_key.setdefault((str(r.get('scene_id')), r.get('arm')), []).append(
                r.get('outcome', ''))
        if self._regime == 'lidar':
            # one row per distinct arrangement label seen in the manifest
            row_keys, seen = [], set()
            for r in rows:
                s = str(r.get('scene_id'))
                if s and s not in seen:
                    seen.add(s); row_keys.append(s)
            headers = row_keys
        else:
            row_keys = [str(i) for i in range(self._scene.count())]
            headers = [f'scene {i}' for i in range(self._scene.count())]
        self._table.setRowCount(len(row_keys))
        self._table.setVerticalHeaderLabels(headers)
        palette = {'reached': '#1f5130', 'dnf': '#6b5320', 'collision': '#6b2020',
                   'aborted': '#4a4a4a'}
        for rr, rk in enumerate(row_keys):
            for cc, arm in enumerate(arms):
                outs = by_key.get((rk, arm), [])
                it = QTableWidgetItem(','.join(o[:3] for o in outs if o) if outs else '')
                if outs:
                    it.setBackground(QColor(palette.get(outs[-1], '#23262b')))
                self._table.setItem(rr, cc, it)
        self._cov_lbl.setText(f'{len(rows)} rows · {self._session_dir}')

    def _on_zip(self):
        d = self._session_dir
        if not os.path.isdir(d):
            self._log_line(f'nothing to zip -- {d} does not exist yet')
            return
        out = shutil.make_archive(d.rstrip('/'), 'zip',
                                  root_dir=os.path.dirname(d),
                                  base_dir=os.path.basename(d))
        self._log_line(f'zipped session dir -> {out}')

    def _on_apply_gamma(self):
        v = round(self._spin_gamma.value(), 3)
        if not self._guard_ok(f'cbf_gamma_obs={v}'):
            return
        self._log_line(f'-> ad-hoc: set cbf_gamma_obs = {v} (live); runs tagged arm=adhoc')
        self._node.set_params({'cbf_gamma_obs': v})

    # ROS slots
    def _on_params_read(self, d):
        if not self._params_synced:
            self._log_line('params read from CBF node: ' + ', '.join(
                f'{k}={v}' for k, v in d.items()))
        self._params_synced = True

    def _on_params_result(self, ok, msg):
        self._log_line(('OK  ' if ok else 'FAIL ') + msg)

    def _on_status(self, msg):
        armed = (msg.arming_state == _ARMED)
        offb = (msg.nav_state == _OFFBOARD)
        if armed != self._armed:
            self._log_line(f'arming: {"ARMED" if armed else "DISARMED"}')
            if armed:
                self._view.clear_trail()
            self._armed = armed
        self._c_arm.setText(f'ARM: {"ARMED" if armed else "DISARMED"}')
        self._c_arm.setStyleSheet(_chip('', armed).styleSheet())
        self._c_offb.setText(f'OFFBOARD: {"yes" if offb else "no"}')
        self._c_offb.setStyleSheet(_chip('', offb).styleSheet())

    def _on_battery(self, msg):
        self._v['batt'].setText(f'{msg.voltage_v:.2f} V   {msg.remaining * 100:.0f} %')

    def _on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        vv = msg.twist.twist.linear
        self._view.set_pose(p.x, p.y, _yaw_from_quat(q))
        self._v['pos'].setText(f'{p.x:+.2f}  {p.y:+.2f}  {p.z:+.2f}')
        self._v['vel'].setText(f'{math.sqrt(vv.x**2 + vv.y**2 + vv.z**2):.2f}')
        self._odom_timer.restart()
        self._have_odom = True

    def _on_obstacles(self, msg):
        obs = [(m.pose.position.x, m.pose.position.y, 0.5 * m.scale.x) for m in msg.markers]
        self._view.set_detected_obstacles(obs)

    def _on_diag(self, d):
        self._c_feas.setText('QP: feasible' if d.feasible else 'QP: INFEASIBLE')
        self._c_feas.setStyleSheet(_chip('', d.feasible).styleSheet())
        mode = 'PENN (adaptive)' if d.used_penn else 'fixed'
        self._c_mode.setText(f'mode: {mode}')
        self._c_gamma.setText(f'γ: {d.gamma_used:.2f}')
        self._c_goal.setText('goal: REACHED' if d.reached else f'goal: {d.dist_to_goal:.2f} m')
        self._c_goal.setStyleSheet(_chip('', True if d.reached else None).styleSheet())

        self._v['thrust'].setText(f'{d.thrust:.2f}')
        self._v['delta'].setText(f'{d.delta:.4f}')
        self._v['dhat'].setText(f'{d.d_hat.x:+.2f} {d.d_hat.y:+.2f} {d.d_hat.z:+.2f}')
        self._v['dist_goal'].setText(f'{d.dist_to_goal:.2f}')

        self._view.set_goal((d.waypoint.x, d.waypoint.y))
        self._view.set_min_dist(d.min_obstacle_dist)
        self._update_bar(d.min_obstacle_dist)
        self._last_min_dist = (None if d.min_obstacle_dist > 1e8 else d.min_obstacle_dist)

        if self._last_feasible is True and not d.feasible:
            self._log_line(f'QP INFEASIBLE  (slack delta={d.delta:.4f}, gamma={d.gamma_used:.2f})')
        if self._last_feasible is False and d.feasible:
            self._log_line('QP feasible again')
        self._last_feasible = d.feasible
        if d.reached and self._last_reached is False:
            self._log_line(f'goal reached  (dist={d.dist_to_goal:.3f} m)')
        self._last_reached = d.reached

    def _update_bar(self, m):
        if m is None or m > 1e8:
            self._bar.setValue(300)
            self._bar.setFormat('min obstacle dist: (no obstacles)')
            self._alarm.setText('')
            colour = '#3a7d44'
        else:
            self._bar.setFormat('min obstacle dist: %v cm')
            self._bar.setValue(max(0, min(300, int(m * 100))))
            if m < _ALARM_M:
                colour = '#b02a2a'
                self._alarm.setText(f'⚠  OBSTACLE {m:.2f} m')
            elif m < _WARN_M:
                colour = '#b5892a'
                self._alarm.setText('')
            else:
                colour = '#3a7d44'
                self._alarm.setText('')
        self._bar.setStyleSheet(
            f'QProgressBar{{border:1px solid #444;border-radius:3px;text-align:center;}}'
            f'QProgressBar::chunk{{background:{colour};}}')

    # timers
    def _on_ui_tick(self):
        if self._have_odom:
            age = self._odom_timer.elapsed() / 1000.0
            self._v['odom_age'].setText(f'{age:.1f}')
            _set_style(self._v['odom_age'], 'color:#ff6b6b;' if age > 1.0 else '')
        else:
            self._v['odom_age'].setText('no odom')
            _set_style(self._v['odom_age'], 'color:#ff6b6b;')

        if not self._params_synced and self._param_poll.elapsed() > 1000:
            self._param_poll.restart()
            if self._node.params_service_ready():
                self._node.refresh_params()

    def _slow_tick(self):
        names = self._node.graph_node_names()
        active = {nid for nid, _s in _HEALTH_BY_REGIME.get(self._regime, ())}
        for nid, lbl in self._health.items():
            if nid not in active:
                lbl.setVisible(False)
                continue
            lbl.setVisible(True)
            up = nid in names
            short = dict(_HEALTH_NODES)[nid]
            lbl.setText(('● ' if up else '○ ') + short)
            _set_style(lbl, 'font-family:monospace;font-size:11px;color:%s;'
                       % ('#8fce8f' if up else '#8a8f98'))
        ok_all = True
        for (name, fn), lbl in zip(self._checks, self._check_lbls):
            if name == 'scene push service ready' and self._regime == 'lidar':
                lbl.setText('–  scene push service (n/a for lidar)')
                _set_style(lbl, 'font-family:monospace;font-size:11px;color:#8a8f98;')
                continue
            try:
                ok = bool(fn())
            except Exception:  # noqa: BLE001
                ok = False
            ok_all &= ok
            lbl.setText(('✓  ' if ok else '✗  ') + name)
            _set_style(lbl, 'font-family:monospace;font-size:11px;color:%s;'
                       % ('#8fce8f' if ok else '#d08a5a'))
        self._ready.setText('READY' if ok_all else 'NOT READY')
        _set_style(self._ready, 'font-weight:bold;color:%s;'
                   % ('#8fce8f' if ok_all else '#d08a5a'))

    def shutdown(self):
        if self._bag.running():
            self._bag.stop()
        if self._launcher.running():
            self._launcher.stop()


def main(argv=None):
    argv = argv if argv is not None else sys.argv
    rclpy.init(args=argv)
    node = TelemetryNode()
    spin = RosSpinThread(node)
    spin.start()

    app = QApplication([argv[0]])
    # Dark theme via Fusion + a QPalette rather than an app-wide `QWidget{...}` style sheet.
    app.setStyle('Fusion')
    # Baseline font bump -- a QFont-level change like the palette below, not a blanket QWidget{...}
    # style sheet, so it stays on the fast paint path.
    _font = app.font()
    _font.setPointSize(max(_font.pointSize(), 10))
    app.setFont(_font)
    _pal = QPalette()
    _bg, _fg, _base = QColor('#181a1e'), QColor('#dfe3e8'), QColor('#101215')
    _pal.setColor(QPalette.Window, _bg)
    _pal.setColor(QPalette.WindowText, _fg)
    _pal.setColor(QPalette.Base, _base)
    _pal.setColor(QPalette.AlternateBase, QColor('#20242a'))
    _pal.setColor(QPalette.Text, _fg)
    _pal.setColor(QPalette.Button, QColor('#2a2f36'))
    _pal.setColor(QPalette.ButtonText, _fg)
    _pal.setColor(QPalette.ToolTipBase, _bg)
    _pal.setColor(QPalette.ToolTipText, _fg)
    _pal.setColor(QPalette.Highlight, QColor('#1c6b84'))
    _pal.setColor(QPalette.HighlightedText, QColor('#eafcff'))
    for _grp in (QPalette.Disabled,):
        _pal.setColor(_grp, QPalette.Text, QColor('#6b7178'))
        _pal.setColor(_grp, QPalette.ButtonText, QColor('#6b7178'))
        _pal.setColor(_grp, QPalette.WindowText, QColor('#6b7178'))
    app.setPalette(_pal)
    app.setStyleSheet(
        'QGroupBox{border:1px solid #333;margin-top:8px;}'
        'QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 3px;}'
        # One rule so every plain button (connect / apply live / zip / dialog OK) shares the same
        # proportions instead of Fusion's tight default padding -- _btn_run keeps its own bigger, …
        'QPushButton{padding:5px 14px;min-height:20px;}'
        'QComboBox{padding:3px 6px;min-height:20px;}'
        'QLineEdit,QDoubleSpinBox{padding:3px 4px;min-height:20px;}')
    win = Console(node)
    win.setWindowTitle('ARC Operator Console')

    # Open at the layout's natural size but never larger than 90% of the screen; if even the
    # widget's own minimum doesn't fit, maximise instead.
    avail = app.primaryScreen().availableGeometry()
    hint = win.sizeHint()
    cap_w, cap_h = int(avail.width() * 0.9), int(avail.height() * 0.9)
    w = min(hint.width(), cap_w)
    h = min(hint.height(), cap_h)
    if win.minimumSizeHint().width() > cap_w or win.minimumSizeHint().height() > cap_h:
        win.showMaximized()
    else:
        win.resize(w, h)
        win.move(avail.x() + (avail.width() - w) // 2,
                 avail.y() + (avail.height() - h) // 2)
        win.show()
    rc = 1
    try:
        rc = app.exec_()
    finally:
        try:
            win.shutdown()
        except Exception:  # noqa: BLE001
            pass
        spin.stop()
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(rc)


if __name__ == '__main__':
    main()
