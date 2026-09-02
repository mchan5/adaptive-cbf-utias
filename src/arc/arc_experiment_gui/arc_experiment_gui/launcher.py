"""Subprocess orchestration for the stack and for rosbag recording.

Both are plain child processes in their own process group so the whole tree
can be signalled at once. `ros2 launch` gets SIGINT, then SIGKILL as a
backstop, then a pkill sweep of the companion-side leaf processes that
`ros2 launch` is known not to always reap. `ros2 bag record` gets SIGINT
only -- it needs it to finalise metadata.yaml.

The console can bring up either regime's stack (see `_TARGETS`):
  * hardware -- cbf_rate_arc_hardware_launch.py. MicroXRCEAgent (over
    serial), the Livox driver, and the OptiTrack/mocap driver are
    operator-managed and are NOT started or killed here.
  * sitl -- cbf_rate_arc_sitl_test_launch.py, which starts PX4 SITL, the
    gz_x500 world, and MicroXRCEAgent-over-UDP itself, so the SITL sweep
    list has to reap those too. The px4 match is a FULL PATH
    (px4_sitl_default/bin/px4), never a bare 'px4' -- taken verbatim from
    the proven cleanup() in
    penn_gat_training/tools/run_sitl_repeat_battery.sh.

Nothing here is in the control path; it just starts and stops things.
"""

import os
import signal
import subprocess
import threading
import time

from PyQt5.QtCore import QObject, pyqtSignal

# Per-regime launch descriptor. `leaf_procs` is the `pkill -9 -f` sweep list
# applied on stop() in case `ros2 launch` doesn't reap its children.
_TARGETS = {
    'hardware': {
        'launch_file': 'cbf_rate_arc_hardware_launch.py',
        # Excludes MicroXRCEAgent (serial), the Livox driver, and the mocap
        # driver -- all operator-managed.
        'leaf_procs': (
            'synthetic_obstacle_publisher', 'lidar_obstacle_publisher',
            'autopilot_sv_cbf_rate_node', 'mocap_odom_bridge_node',
            'operator_arm_node', 'geofence_monitor_node',
            'ground_station_stub_node',
        ),
    },
    'sitl': {
        'launch_file': 'cbf_rate_arc_sitl_test_launch.py',
        'leaf_procs': (
            'synthetic_obstacle_publisher', 'autopilot_sv_cbf_rate_node',
            'ground_station_stub_node', 'px4_odom_bridge_node',
            'operator_arm_node', 'geofence_monitor_node',
            # SITL infra this launch starts itself:
            'px4_sitl_default/bin/px4', 'gz sim --verbose', 'MicroXRCEAgent',
        ),
    },
}

# Back-compat alias -- some call sites / tests still import this name.
_HW_LEAF_PROCS = _TARGETS['hardware']['leaf_procs']


class _Proc(QObject):
    line = pyqtSignal(str)
    state = pyqtSignal(str)   # 'starting' | 'running' | 'stopping' | 'stopped'

    def __init__(self):
        super().__init__()
        self._p = None
        self._reader = None

    def running(self):
        return self._p is not None and self._p.poll() is None

    def _spawn(self, argv, env=None):
        self.state.emit('starting')
        self._p = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, start_new_session=True,
            env={**os.environ, **(env or {})})
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()
        self.state.emit('running')

    def _pump(self):
        try:
            for ln in self._p.stdout:
                self.line.emit(ln.rstrip())
        except Exception:  # noqa: BLE001
            pass
        rc = self._p.wait()
        self.line.emit(f'[process exited rc={rc}]')
        self.state.emit('stopped')

    def _signal_group(self, sig):
        if self._p and self._p.poll() is None:
            try:
                os.killpg(os.getpgid(self._p.pid), sig)
            except ProcessLookupError:
                pass


class StackLauncher(_Proc):
    """`ros2 launch` of a CBF companion stack -- hardware or SITL.

    `target` selects the launch file and the teardown sweep list (see
    `_TARGETS`). `obstacle_source` picks synthetic (manual, fed a frozen
    scene JSON) vs live LiDAR clustering -- the same vocabulary in both
    regimes now that the two launch files take symmetric arguments.

    For SITL the console drives an unattended flight: `arm_mode:=auto` so
    ground_station_stub_node arms the vehicle, with `publish_goal:=false` so
    the stub does NOT fight the GUI for goal_waypoint. Hardware keeps the
    launch default (`arm_mode:=operator`) -- the pilot arms.
    """

    def __init__(self):
        super().__init__()
        self._target = 'hardware'

    def start(self, scene_path='', obstacle_source='manual',
              launch_rviz=False, extra_env=None, target='hardware',
              cylinder_barrier=False):
        if self.running():
            self.line.emit('[stack already running]')
            return
        if target not in _TARGETS:
            self.line.emit(f'[unknown target {target!r}]')
            return
        self._target = target
        spec = _TARGETS[target]
        argv = ['ros2', 'launch', 'single_vehicle_cbf_rate_arc', spec['launch_file'],
                f'launch_rviz:={"true" if launch_rviz else "false"}',
                f'obstacle_source:={obstacle_source}']
        if obstacle_source == 'manual' and scene_path:
            argv.append(f'obstacle_file:={scene_path}')
        # cbf_cylinder_barrier is a structural param (read once at node
        # construction), so it can only be set here at launch, not live.
        # Both the SITL and hardware launch files declare it.
        argv.append(
            f'cbf_cylinder_barrier:={"true" if cylinder_barrier else "false"}')
        if target == 'sitl':
            argv += ['arm_mode:=auto', 'publish_goal:=false']
        self.line.emit('$ ' + ' '.join(argv))
        self._spawn(argv, env=extra_env)

    def stop(self):
        if not self.running():
            self._pkill_leaves()
            self.state.emit('stopped')
            return
        self.state.emit('stopping')
        self._signal_group(signal.SIGINT)
        for _ in range(80):
            if self._p.poll() is not None:
                break
            time.sleep(0.1)
        if self._p.poll() is None:
            self._signal_group(signal.SIGKILL)
        self._pkill_leaves()
        self.state.emit('stopped')

    def _pkill_leaves(self):
        for pat in _TARGETS[self._target]['leaf_procs']:
            subprocess.run(['pkill', '-9', '-f', pat],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class BagRecorder(_Proc):
    """`ros2 bag record` for one trial."""

    TOPICS = [
        '/{p}/cbf/diagnostics',
        '/{p}/state_estimator/local_position/odom',
        '/arc/obstacles',
        '/{p}/fmu/out/vehicle_status',
        '/{p}/fmu/in/vehicle_rates_setpoint',
        '/{p}/goal_waypoint',
        '/{p}/fmu/out/battery_status',
        '/tf', '/tf_static',
    ]

    def start(self, out_dir, uav_prefix='uav_0', extra_topics=None):
        if self.running():
            self.line.emit('[bag already recording]')
            return
        topics = [t.format(p=uav_prefix) for t in self.TOPICS]
        topics += list(extra_topics or [])
        argv = ['ros2', 'bag', 'record', '-o', out_dir] + topics
        self.line.emit('$ ' + ' '.join(argv))
        self._spawn(argv)

    def stop(self):
        if not self.running():
            self.state.emit('stopped')
            return
        self.state.emit('stopping')
        self._signal_group(signal.SIGINT)
        for _ in range(60):
            if self._p.poll() is not None:
                break
            time.sleep(0.1)
        if self._p.poll() is None:
            self._signal_group(signal.SIGKILL)
        self.state.emit('stopped')
