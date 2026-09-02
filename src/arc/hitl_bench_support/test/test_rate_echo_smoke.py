"""Offline smoke test for hitl_rate_echo_node -- no Pixhawk, no SITL.

Spins the echo node plus a harness node in one process and checks the four
parts of its safety contract:

  * gate    -- nothing is forwarded until bench_enable goes true
  * restamp -- the forwarded timestamp is this machine's wall clock, not the
               sim timestamp that came in
  * clamp   -- normalised thrust magnitude is capped at thrust_clamp_norm
  * deadman -- when the sim rate stream stops, a zero-rate hold is published

Run:  python3 -m pytest hitl_bench_support/test/test_rate_echo_smoke.py -v
"""

import time

import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from px4_msgs.msg import OffboardControlMode, VehicleRatesSetpoint
from std_msgs.msg import Bool

from hitl_bench_support.hitl_rate_echo_node import HitlRateEchoNode, _px4_qos

SIM = 'uav_0'
BENCH = 'uav_bench'
CLAMP = 0.15
DEADMAN = 0.1


class _Harness(Node):
    """Stands in for SITL (publishes sim rates) and the bench board (records
    what would have gone to the real FMU)."""

    def __init__(self):
        super().__init__('hitl_smoke_harness')
        qos = _px4_qos()
        self.rate_pub = self.create_publisher(
            VehicleRatesSetpoint, f'/{SIM}/fmu/in/vehicle_rates_setpoint', qos)
        self.ocm_pub = self.create_publisher(
            OffboardControlMode, f'/{SIM}/fmu/in/offboard_control_mode', qos)
        self.enable_pub = self.create_publisher(
            Bool, f'/{BENCH}/bench_enable', 10)
        self.rx_rates = []
        self.rx_ocm = []
        self.create_subscription(
            VehicleRatesSetpoint, f'/{BENCH}/fmu/in/vehicle_rates_setpoint',
            self.rx_rates.append, qos)
        self.create_subscription(
            OffboardControlMode, f'/{BENCH}/fmu/in/offboard_control_mode',
            self.rx_ocm.append, qos)

    def send_rate(self, sim_timestamp, thrust_z):
        m = VehicleRatesSetpoint()
        m.timestamp = int(sim_timestamp)
        m.roll, m.pitch, m.yaw = 0.1, 0.2, 0.3
        m.thrust_body = [0.0, 0.0, float(thrust_z)]
        self.rate_pub.publish(m)
        ocm = OffboardControlMode()
        ocm.body_rate = True
        self.ocm_pub.publish(ocm)

    def set_enable(self, val):
        self.enable_pub.publish(Bool(data=bool(val)))


@pytest.fixture
def rig():
    rclpy.init()
    echo = HitlRateEchoNode()
    # Tighten timing so the test runs in well under a second.
    for name, value in (('thrust_clamp_norm', CLAMP),
                        ('deadman_timeout_sec', DEADMAN),
                        ('heartbeat_hz', 100.0)):
        echo.set_parameters([rclpy.parameter.Parameter(name, value=value)])
    # set_parameters after construction does not re-run __init__; refresh the
    # cached copies the node read once at start.
    echo._clamp = CLAMP
    echo._deadman = DEADMAN
    harness = _Harness()
    ex = SingleThreadedExecutor()
    ex.add_node(echo)
    ex.add_node(harness)
    yield echo, harness, ex
    ex.shutdown()
    echo.destroy_node()
    harness.destroy_node()
    rclpy.shutdown()


def _pump(ex, seconds, hz=200):
    end = time.time() + seconds
    while time.time() < end:
        ex.spin_once(timeout_sec=1.0 / hz)


def test_gate_blocks_until_enabled(rig):
    echo, harness, ex = rig
    _pump(ex, 0.2)
    for _ in range(10):
        harness.send_rate(sim_timestamp=1000, thrust_z=-0.05)
        _pump(ex, 0.02)
    assert harness.rx_rates == [], 'forwarded rates before bench_enable'

    harness.set_enable(True)
    _pump(ex, 0.1)
    for _ in range(10):
        harness.send_rate(sim_timestamp=2000, thrust_z=-0.05)
        _pump(ex, 0.02)
    assert harness.rx_rates, 'nothing forwarded after bench_enable rising edge'


def test_restamp_and_clamp(rig):
    echo, harness, ex = rig
    harness.set_enable(True)
    _pump(ex, 0.1)
    sim_ts = 4242
    for _ in range(6):
        harness.send_rate(sim_timestamp=sim_ts, thrust_z=-1.0)
        _pump(ex, 0.03)
    fwd = [m for m in harness.rx_rates
           if m.roll == pytest.approx(0.1) and abs(m.pitch - 0.2) < 1e-6]
    assert fwd, 'no forwarded command carrying the harness payload'
    m = fwd[-1]
    tb = [float(v) for v in m.thrust_body]
    # re-stamped with wall-clock microseconds, not the sim timestamp
    now_us = time.time() * 1e6
    assert m.timestamp != sim_ts
    assert abs(m.timestamp - now_us) < 5e6, (m.timestamp, now_us)
    # full-throttle demand clamped to the configured magnitude
    # ROS float32 fields, so compare with a float32-sized tolerance
    assert tb[2] == pytest.approx(-CLAMP, abs=1e-6)
    assert min(tb) >= -CLAMP - 1e-6
    assert max(tb) <= CLAMP + 1e-6


def test_deadman_holds_zero_when_sim_stops(rig):
    echo, harness, ex = rig
    harness.set_enable(True)
    _pump(ex, 0.1)
    for _ in range(6):
        harness.send_rate(sim_timestamp=5000, thrust_z=-0.12)
        _pump(ex, 0.03)
    n_before = len(harness.rx_rates)
    # stop feeding sim rates; within deadman + a couple of heartbeats the
    # node should emit a zero-rate safe hold
    _pump(ex, DEADMAN + 0.15)
    holds = harness.rx_rates[n_before:]
    assert holds, 'no output at all after the sim stream stopped'
    last = holds[-1]
    assert (last.roll, last.pitch, last.yaw) == (0.0, 0.0, 0.0)
    assert [float(v) for v in last.thrust_body] == [0.0, 0.0, 0.0]
    assert echo._deadman_latched
