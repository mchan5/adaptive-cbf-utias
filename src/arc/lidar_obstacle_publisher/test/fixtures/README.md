# Bench replay fixture

`test/test_bag_replay.py` replays a recorded Livox bag through
`LidarObstaclePublisherNode` and checks detections against measured ground
truth. It is the integration coverage SITL can't provide (real point clouds,
real transform, real clustering + tracking).

Until data exists the test **skips** (`bench_replay.yaml` ships with
`bag_path: null`).

## Recording (lab time)

Prereqs, both live and confirmed with `ros2 topic hz`:
- Livox driver publishing `/livox/lidar` (`xfer_format:=0`, i.e. real
  `sensor_msgs/PointCloud2` -- see the node docstring)
- an odom source publishing `/state_estimator/local_position/odom` in
  world/ENU (mocap bridge, or the onboard EKF) -- the sensor must be rigidly
  attached to the body that odom describes

Then:

```
tools/record_bench_bag.sh            # Ctrl-C after ~20-40 s
```

During the take: one box/pole at a **measured** world position, kept inside
`max_range_m` and the `[z_min_m, z_max_m]` band, swept through the FOV for
several seconds by moving the vehicle.

## Filling in `bench_replay.yaml`

- `bag_path`: the recorded directory name (kept in this folder; the bag
  itself is gitignored, the yaml is committed)
- `params.sensor_offset_xyz` / `sensor_offset_rpy_deg`: the mount offset in
  effect during the recording
- `obstacles[].center_xyz` / `radius_m`: measured, in the odom frame
- `expect.*`: tolerances -- start with the shipped values, tighten once you
  see real numbers

## Running

```
colcon test --packages-select lidar_obstacle_publisher
# or, from the package root:
pytest test/test_bag_replay.py -q
```

A red test here means the transform, clustering, or tracker regressed
against real data -- check the mount offset first.
