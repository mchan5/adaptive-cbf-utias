"""SITL vs hardware parameter-file parity.

The sim (`params_single_vehicle_cbf_rate_arc.yaml`) and hardware
(`..._hardware.yaml`) parameter files are meant to differ ONLY in a small,
explicit set of regime-specific values -- vehicle model, venue geometry,
conservatism, and PENN calibration. Everything else (control-law gains, QP
bounds, the nominal controller, the infeasible-brake fallback) must be
identical so a SITL rehearsal actually exercises the config that ships on
hardware.

This test pins that contract:

  * both files expose the SAME set of parameter keys (a key present in one
    and absent from the other silently falls back to the C++
    declare_parameter default on the side that omits it -- exactly the
    'frozen config' drift HARDWARE_COMMANDS_20260828.md tries to grep for);
  * every value is a real scalar, not a string (guards the 1.0e4 ->
    string-under-safe_load foot-gun);
  * every key NOT in the allow-to-differ list holds an equal value in both
    files.

If you deliberately introduce a new regime difference, add the key to
ALLOWED_TO_DIFFER below with a one-line reason. That edit is the record.
"""

import math
import os

import pytest
import yaml

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, 'config')
_SITL = os.path.join(_CONFIG_DIR, 'params_single_vehicle_cbf_rate_arc.yaml')
_HW = os.path.join(_CONFIG_DIR, 'params_single_vehicle_cbf_rate_arc_hardware.yaml')

_NODE_KEY = '/**/autopilot_sv_cbf_rate_node'

# Keys allowed to hold different values in the two files, each with the
# reason. Everything else must match.
ALLOWED_TO_DIFFER = {
    'waypoint_x': 'venue geometry',
    'waypoint_y': 'venue geometry',
    'waypoint_z': 'venue geometry (real cylinders are 0.75 m tall)',
    'vehicle_mass': 'real vehicle vs Gazebo x500',
    'cbf_gamma_obs': 'hardware starts more conservative',
    'uncertified_v_cap_enabled': 'hardware starts more conservative',
    'uncertified_v_cap': 'hardware starts more conservative',
    'penn_enabled': 'adaptive PENN off on hardware until spot-checked',
    'penn_gamma_min': 'hardware PENN checkpoint calibration',
    'penn_gamma_max': 'hardware PENN checkpoint calibration',
    'epistemic_threshold': 'hardware PENN checkpoint calibration',
    'cvar_boundary': 'hardware PENN checkpoint calibration',
    'lcb_k': 'hardware PENN checkpoint calibration',
}

# The one key whose value is legitimately a string.
_STRING_KEYS = {'penn_model_path'}


def _load(path):
    with open(path) as f:
        doc = yaml.safe_load(f)
    return doc[_NODE_KEY]['ros__parameters']


@pytest.fixture(scope='module')
def params():
    return _load(_SITL), _load(_HW)


def test_key_sets_identical(params):
    sitl, hw = params
    only_sitl = sorted(set(sitl) - set(hw))
    only_hw = sorted(set(hw) - set(sitl))
    assert not only_sitl, f'keys only in the SITL yaml: {only_sitl}'
    assert not only_hw, f'keys only in the hardware yaml: {only_hw}'


def test_no_stringly_typed_scalars(params):
    for name, table in zip(('sitl', 'hardware'), params):
        for key, val in table.items():
            if key in _STRING_KEYS:
                assert isinstance(val, str), f'{name}:{key} should be a string path'
                continue
            assert isinstance(val, (int, float, bool)) and not isinstance(val, str), (
                f'{name}:{key} = {val!r} is not a numeric/bool scalar '
                f'(YAML like 1.0e4 parses as a string under safe_load)'
            )


def test_shared_keys_have_equal_values(params):
    sitl, hw = params
    mismatched = {}
    for key in set(sitl) & set(hw):
        if key in ALLOWED_TO_DIFFER:
            continue
        a, b = sitl[key], hw[key]
        if isinstance(a, float) and isinstance(b, float):
            if math.isclose(a, b, rel_tol=0.0, abs_tol=0.0):
                continue
        elif a == b:
            continue
        mismatched[key] = (a, b)
    assert not mismatched, (
        'shared keys differ but are not in ALLOWED_TO_DIFFER '
        f'(sitl, hardware): {mismatched}'
    )
