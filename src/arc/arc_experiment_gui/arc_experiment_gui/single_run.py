"""One-press single-trial runner for the operator console.

The 4-tab GUI only recorded a bag on the campaign path (`TrialRunner`, which
walks a whole plan). The redesigned single-screen console has one START RUN
button that runs exactly one trial and appends one row to a per-session
`manifest.json`. This is that state machine -- a trimmed `TrialRunner`:

  push arm params -> push scene obstacles -> start rosbag ->
  move the setpoint to the pre-established goal -> wait `reached` /
  timeout / abort -> stop rosbag -> operator confirms outcome ->
  append manifest row.

Return-to-start is regime-conditional. On hardware the pilot positions the
vehicle and flips to OFFBOARD on the RC, so START RUN goes straight to
recording and just moves the goal_waypoint to GOAL_XYZ. In SITL there is no
pilot, so after the first trial the vehicle is parked on the goal and the
next run would read "reached" instantly; SITL therefore gets a leading
"fly back to START_XYZ, wait until we are there" leg before the bag starts.

Nothing here is in the control loop; it only sets parameters, publishes a
goal, and records a bag.
"""

import os
import time

from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from arc_experiment_gui.campaign import Manifest
from arc_experiment_gui.runner import (
    GOAL_XYZ, GOAL_TIMEOUT, BAG_SPINUP, START_XYZ, RETURN_TOL, RETURN_TIMEOUT)

SCENE_SPINUP = 1.5   # s to let the obstacle publisher pick up the new param


class SingleRunner(QObject):
    progress = pyqtSignal(str)
    state = pyqtSignal(str)                 # phase name
    awaiting_outcome = pyqtSignal(object)  # {'suggested': str, 'row': dict}
    trial_written = pyqtSignal(object)     # the full manifest row
    run_done = pyqtSignal()

    def __init__(self, node, bag):
        super().__init__()
        self._node = node
        self._bag = bag
        self._phase = 'idle'
        self._t_phase = 0.0
        self._t_goal0 = 0.0
        self._buf = []
        self._last_diag = None
        node.diag.connect(self._on_diag)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    # public
    def busy(self):
        return self._phase not in ('idle', 'done')

    def start(self, session_dir, scene_id, arm, arm_cfg, scene_flat,
              regime, git_state, record=True):
        if self.busy():
            self.progress.emit('run already in progress')
            return
        os.makedirs(session_dir, exist_ok=True)
        self._dir = session_dir
        self._sid = scene_id                # int (synthetic), str label (lidar), None (ad-hoc)
        self._arm = arm
        self._cfg = dict(arm_cfg or {})
        self._scene_flat = scene_flat       # list, or None to skip the step
        self._regime = regime
        # SITL has no pilot to reposition the vehicle between trials, so it
        # flies itself back to START_XYZ before the bag starts. Hardware /
        # synthetic / lidar skip straight to recording.
        self._return_first = (regime == 'sitl')
        self._git = git_state
        self._record = bool(record)
        self._manifest = Manifest(session_dir)
        self._k = 1 + sum(
            1 for r in self._manifest.rows
            if str(r.get('scene_id')) == str(scene_id) and r.get('arm') == arm)
        self._buf = []
        self._enter('set_params')
        self._timer.start(250)

    def abort(self):
        if not self.busy():
            return
        self.progress.emit('ABORT requested')
        if self._bag.running():
            self._bag.stop()
        self._stage('aborted', time.time() - self._t_goal0 if self._t_goal0 else 0.0)

    def submit_outcome(self, outcome, notes=''):
        if self._phase != 'await_outcome':
            return
        row = dict(self._pending_row)
        row['outcome'] = outcome
        row['notes'] = notes
        full = self._manifest.append(row)
        self.trial_written.emit(full)
        self.progress.emit(
            f'wrote trial: {full["scene_id"]}/{full["arm"]} -> {outcome}')
        self._enter('done')
        self._timer.stop()
        self.run_done.emit()

    # machine
    def _enter(self, phase):
        self._phase = phase
        self._t_phase = time.time()
        self.state.emit(phase)

    def _age(self):
        return time.time() - self._t_phase

    def _after_scene(self):
        return 'return_to_start' if self._return_first else 'record_start'

    def _on_diag(self, d):
        self._last_diag = d
        if self._phase in ('goto_goal', 'wait_reached'):
            self._buf.append((time.time() - self._t_goal0, d.gamma_used,
                              bool(d.used_penn), d.min_obstacle_dist))

    def _bag_dir(self):
        if self._sid is None:
            tag = 'adhoc'
        elif isinstance(self._sid, int):
            tag = f'scene{self._sid:02d}'
        else:
            tag = str(self._sid)                 # lidar arrangement label
        return os.path.join(self._dir, f'{tag}_{self._arm}_trial{self._k}')

    def _tick(self):
        ph = self._phase

        if ph == 'set_params':
            self.progress.emit(f'arm {self._arm}: setting {self._cfg or "(none)"}')
            if self._cfg:
                self._node.set_params(self._cfg)
            self._enter('set_scene')
            return

        if ph == 'set_scene':
            if self._scene_flat is None:
                self._enter(self._after_scene())
                return
            if self._age() < 0.4:
                return
            self._node.set_scene_obstacles(self._scene_flat)
            if self._age() > SCENE_SPINUP:
                self._enter(self._after_scene())
            return

        if ph == 'return_to_start':
            # No stub is republishing goal_waypoint in SITL (publish_goal:=false),
            # so keep nudging it to START every tick. diag.dist_to_goal tracks
            # the active waypoint, which we have just moved to START; give it a
            # beat to propagate before trusting the distance.
            self._node.publish_goal(*START_XYZ)
            if self._age() < 1.0:
                return
            d = self._last_diag
            dist = None if d is None else d.dist_to_goal
            if dist is not None and dist < RETURN_TOL:
                self.progress.emit(f'back at start (d={dist:.2f} m)')
                self._enter('record_start')
            elif self._age() > RETURN_TIMEOUT:
                self.progress.emit(
                    f'return-to-start timed out ({RETURN_TIMEOUT:.0f} s) -- recording anyway')
                self._enter('record_start')
            return

        if ph == 'record_start':
            if self._record:
                extra = ['/livox/lidar'] if self._regime == 'lidar' else None
                self._bag.start(self._bag_dir(), self._node.uav_prefix, extra)
            else:
                self.progress.emit('recording disabled for this run -- no bag')
            self._buf = []
            self._enter('goto_goal')
            return

        if ph == 'goto_goal':
            if self._age() < BAG_SPINUP:
                return
            self._node.publish_goal(*GOAL_XYZ)
            if self._age() < BAG_SPINUP + 1.5:
                return
            self._t_goal0 = time.time()
            self.progress.emit('mission goal published -- timing')
            self._enter('wait_reached')
            return

        if ph == 'wait_reached':
            self._node.publish_goal(*GOAL_XYZ)
            d = self._last_diag
            t_goal = time.time() - self._t_goal0
            if bool(d and d.reached):
                self.progress.emit(f'REACHED in {t_goal:.1f} s')
                self._stage('reached', t_goal)
            elif t_goal > GOAL_TIMEOUT:
                self.progress.emit(f'DNF (>{GOAL_TIMEOUT:.0f} s)')
                self._stage('dnf', t_goal)
            return

    # staging
    def _min_obs(self):
        vals = [m for (_t, _g, _p, m) in self._buf if m < 1e8]
        return min(vals) if vals else float('nan')

    def _penn_duty(self):
        if not self._buf:
            return float('nan')
        return sum(1 for (_t, _g, p, _m) in self._buf if p) / len(self._buf)

    def _stage(self, suggested, t_goal):
        if self._bag.running():
            self._bag.stop()
        h, dirty = self._git
        self._pending_row = {
            'scene_id': '' if self._sid is None else self._sid,
            'arm': self._arm, 'trial_index': self._k,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'regime': self._regime, 'git_hash': h, 'config_dirty': dirty,
            'outcome': '', 't_goal': round(t_goal, 2),
            'min_obstacle_dist': round(self._min_obs(), 3),
            'penn_duty_cycle': round(self._penn_duty(), 3),
            'bag_path': (os.path.relpath(self._bag_dir(), self._dir)
                         if self._record else ''),
            'notes': '',
        }
        self._enter('await_outcome')
        self.awaiting_outcome.emit(
            {'suggested': suggested, 'row': dict(self._pending_row)})
