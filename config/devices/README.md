# Per-device configuration

The desktop, laptop and Jetson share one `master` branch. Anything whose
correct value differs per machine lives here, **not** in a tracked source file.

## How it works

1. `scripts/setup.sh` runs once per machine. It detects which device it is
   (by hostname), copies the matching `*.env` here to `../../.arc-local.env`
   at the repo root, and appends the resolved `ARC_ROOT`.
2. `.arc-local.env` is gitignored. Every script (`build.sh`, `hardware_run.sh`)
   sources it.
3. To add a machine: copy `desktop.env` to `<newhost>.env`, edit, and add the
   hostname match to `scripts/setup.sh`.

## Fields

| Field | Meaning |
|---|---|
| `ARC_DEVICE` | short name, echoed by scripts for sanity |
| `ROS_DISTRO_EXPECTED` | `jazzy` or `humble`; `build.sh` warns on mismatch |
| `ARC_REGIME` | `sitl` or `hardware` — selects launch files and odom source |
| `MOTIVE_PC_IP` / `THIS_MACHINE_IP` | OptiTrack/Motive networking (hardware only) |
| `PIXHAWK_SERIAL_DEV` / `PIXHAWK_BAUD` | FMU link (hardware only) |
| `UAV_PREFIX` | must match the rigid-body name in Motive |
| `ARC_HAS_LIVOX` | `1` if a Livox LiDAR is attached |
