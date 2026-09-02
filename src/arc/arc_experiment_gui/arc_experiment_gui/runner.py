"""Per-trial state machine for the campaign runner.

One trial =
  set arm params -> (SITL) fly back to start -> start rosbag ->
  publish mission goal -> wait for `reached` / timeout / abort ->
  stop rosbag -> operator confirms outcome -> append manifest row -> next.

Driven by a 250 ms QTimer plus the live CbfDiagnostics stream. Nothing here
touches the control loop; it only sets parameters, publishes a goal, and
records a bag.
"""

import os
import shlex
import signal
import subprocess
import time

from PyQt5.QtCore import QObject, QTimer, pyqtSignal

# Optional per-trial screen/camera capture (Phase 5, opt-in, hardware demos).
# Set ARC_TRIAL_RECORD_CMD to a command template containing "{out}"; it is run
# at bag-start and SIGINT'd at bag-stop. Example:
#   export ARC_TRIAL_RECORD_CMD='ffmpeg -y -f x11grab -i :0 {out}/screen.mp4'
_RECORD_CMD = os.environ.get('ARC_TRIAL_RECORD_CMD', '')

# Fixed SITL corridor (spawn ~origin, goal hard-coded in the launch stack).
START_XYZ = (0.0, 0.5, 1.5)
GOAL_XYZ = (0.0, 5.5, 1.5)

RETURN_TOL = 0.5      # m, "back at start"
RETURN_TIMEOUT = 30.0
GOAL_TIMEOUT = 45.0   # m, DNF if not reached in this long
BAG_SPINUP = 1.8


class TrialRunner(QObject):
    progress = pyqtSignal(str)
    state = pyqtSignal(str)                 # phase name
    cell_changed = pyqtSignal(object)      # dict: scene_id, trial_index, arm, idx, total
    awaiting_outcome = pyqtSignal(object)  # dict: suggested row, waits for submit_outcome()
    trial_written = pyqtSignal(object)     # dict: the manifest row
    campaign_done = pyqtSignal()

    def __init__(self, node, bag, arms_cfg):
        super().__init__()
        self._node = node
        self._bag = bag
        self._arms_cfg = arms_cfg
        self._plan = None
        self._manifest = None
        self._cells = []
        self._idx = 0
        self._phase = 'idle'
        self._t_phase = 0.0
        self._t_goal0 = 0.0
        self._buf = []            # (t, gamma_used, used_penn, min_obstacle_dist)
        self._last_diag = None
        self._git = ('', False)
        self._rec = None          # optional screen/camera capture subprocess

        node.diag.connect(self._on_diag)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    # public
    def start_campaign(self, plan, manifest, git_state):
        self._plan = plan
        self._manifest = manifest
        self._git = git_state
        done = manifest.done_keys()
        self._cells = [c for c in plan.sequence()
                       if (str(c[0]), c[1], c[2]) not in done]
        self._idx = 0
        self.progress.emit(
            f'campaign: {len(self._cells)} trials to run '
            f'({len(plan.sequence()) - len(self._cells)} already done)')
        if not self._cells:
            self.campaign_done.emit()
            return
        self._enter('set_params')
        self._timer.start(250)

    def abort(self):
        if self._phase in ('idle', 'done'):
            return
        self.progress.emit('ABORT requested')
        if self._bag.running():
            self._bag.stop()
        self._stop_capture()
        self._finish_trial('aborted')

    def submit_outcome(self, outcome, notes=''):
        if self._phase != 'await_outcome':
            return
        row = self._pending_row
        row['outcome'] = outcome
        row['notes'] = notes
        full = self._manifest.append(row)
        self.trial_written.emit(full)
        self.progress.emit(f'wrote trial: {full["scene_id"]}/{full["arm"]} -> {outcome}')
        self._idx += 1
        if self._idx >= len(self._cells):
            self._enter('done')
            self._timer.stop()
            self.campaign_done.emit()
        else:
            self._enter('set_params')

    def paused(self):
        return self._phase == 'await_outcome'

    # machine
    def _enter(self, phase):
        self._phase = phase
        self._t_phase = time.time()
        self.state.emit(phase)

    def _age(self):
        return time.time() - self._t_phase

    def _cur(self):
        sid, k, arm = self._cells[self._idx]
        return sid, k, arm

    def _on_diag(self, d):
        self._last_diag = d
        if self._phase in ('goto_goal', 'wait_reached'):
            self._buf.append((time.time() - self._t_goal0, d.gamma_used,
                              bool(d.used_penn), d.min_obstacle_dist))

    def _dist_to(self, xyz):
        d = self._last_diag
        # diag carries dist_to_goal for the *current* waypoint; for the
        # return-to-start leg we set that waypoint to START, so dist_to_goal
        # is exactly what we want. GOAL leg likewise.
        return None if d is None else d.dist_to_goal

    def _tick(self):
        ph = self._phase
        sid, k, arm = self._cur()

        if ph == 'set_params':
            self.cell_changed.emit({'scene_id': sid, 'trial_index': k, 'arm': arm,
                                    'idx': self._idx, 'total': len(self._cells)})
            cfg = dict(self._arms_cfg.get(arm, {}))
            self.progress.emit(f'[{self._idx + 1}/{len(self._cells)}] scene {sid} '
                               f'trial {k} arm {arm}: setting {cfg}')
            self._node.set_params(cfg)
            self._enter('return_to_start')
            return

        if ph == 'return_to_start':
            if self._age() < 1.0:
                return
            self._node.publish_goal(*START_XYZ)
            if self._age() > 2.5:
                self._enter('await_return')
            return

        if ph == 'await_return':
            self._node.publish_goal(*START_XYZ)
            dist = self._dist_to(START_XYZ)
            if dist is not None and dist < RETURN_TOL:
                self.progress.emit(f'back at start (d={dist:.2f} m)')
                self._enter('record_start')
            elif self._age() > RETURN_TIMEOUT:
                self.progress.emit('return-to-start timed out -- recording anyway')
                self._enter('record_start')
            return

        if ph == 'record_start':
            bag_dir = os.path.join(
                self._plan.root, f'scene{int(sid):02d}_{arm}_trial{k}')
            self._bag_dir = bag_dir
            self._bag.start(bag_dir, self._node.uav_prefix)
            self._start_capture(bag_dir)
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
            reached = bool(d and d.reached)
            t_goal = time.time() - self._t_goal0
            if reached:
                self.progress.emit(f'REACHED in {t_goal:.1f} s')
                self._stop_and_stage('reached', t_goal)
            elif t_goal > GOAL_TIMEOUT:
                self.progress.emit(f'DNF (>{GOAL_TIMEOUT:.0f} s)')
                self._stop_and_stage('dnf', t_goal)
            return

        # await_outcome / idle / done: nothing to do on tick

    def _min_obs(self):
        vals = [m for (_t, _g, _p, m) in self._buf if m < 1e8]
        return min(vals) if vals else float('nan')

    def _penn_duty(self):
        if not self._buf:
            return float('nan')
        return sum(1 for (_t, _g, p, _m) in self._buf if p) / len(self._buf)

    def _start_capture(self, out_dir):
        if not _RECORD_CMD:
            return
        try:
            argv = shlex.split(_RECORD_CMD.replace('{out}', out_dir))
            self._rec = subprocess.Popen(argv, start_new_session=True,
                                         stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL)
            self.progress.emit(f'capture: {argv[0]} started')
        except Exception as exc:  # noqa: BLE001
            self.progress.emit(f'capture failed to start: {exc}')
            self._rec = None

    def _stop_capture(self):
        if self._rec and self._rec.poll() is None:
            try:
                os.killpg(os.getpgid(self._rec.pid), signal.SIGINT)
            except ProcessLookupError:
                pass
        self._rec = None

    def _stop_and_stage(self, suggested, t_goal):
        if self._bag.running():
            self._bag.stop()
        self._stop_capture()
        sid, k, arm = self._cur()
        h, dirty = self._git
        self._pending_row = {
            'scene_id': sid, 'arm': arm, 'trial_index': k,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'regime': self._plan.regime, 'git_hash': h, 'config_dirty': dirty,
            'outcome': '', 't_goal': round(t_goal, 2),
            'min_obstacle_dist': round(self._min_obs(), 3),
            'penn_duty_cycle': round(self._penn_duty(), 3),
            'bag_path': os.path.relpath(self._bag_dir, self._plan.root),
            'notes': '',
        }
        self._enter('await_outcome')
        self.awaiting_outcome.emit({'suggested': suggested, 'row': dict(self._pending_row)})

    def _finish_trial(self, outcome):
        # abort path: stage a row and auto-submit
        if self._phase == 'await_outcome':
            self.submit_outcome(outcome, 'aborted by operator')
            return
        sid, k, arm = self._cur()
        h, dirty = self._git
        t_goal = time.time() - self._t_goal0 if self._t_goal0 else 0.0
        self._pending_row = {
            'scene_id': sid, 'arm': arm, 'trial_index': k,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'regime': self._plan.regime, 'git_hash': h, 'config_dirty': dirty,
            'outcome': '', 't_goal': round(t_goal, 2),
            'min_obstacle_dist': round(self._min_obs(), 3),
            'penn_duty_cycle': round(self._penn_duty(), 3),
            'bag_path': os.path.relpath(getattr(self, '_bag_dir', ''), self._plan.root)
            if getattr(self, '_bag_dir', '') else '',
            'notes': '',
        }
        self._enter('await_outcome')
        self.submit_outcome(outcome, 'aborted by operator')
