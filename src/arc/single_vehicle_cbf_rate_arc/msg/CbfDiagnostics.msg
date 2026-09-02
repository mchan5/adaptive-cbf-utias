# Per-tick diagnostics from the body-rate CBF-QP node
# (autopilot_sv_cbf_rate_node). Published off to the side for visualization
# / experiment logging only -- nothing in the control loop consumes it.
# BEST_EFFORT, shallow depth: a slow subscriber must never back-pressure the
# 100 Hz control node.
std_msgs/Header header

# --- gamma / mode ---
float64 gamma_used            # gamma actually applied to the HOCBF this tick
bool used_penn                # true = PENN (adaptive) is live and chose it;
                              # false = fixed cbf_gamma_obs
float64 gamma_fixed_param     # current cbf_gamma_obs parameter value, shown
                              # even while adaptive (it is the fallback gamma)
float64 penn_gamma_min        # adaptive selection range, low
float64 penn_gamma_max        # adaptive selection range, high

# --- QP health ---
bool feasible                 # QP returned a feasible solution this tick
float64 delta                 # QP slack magnitude (0 = hard constraints met
                              # without relaxation)
uint32 certified_count        # obstacles passing the entry-certification gate
uint32 uncertified_count      # obstacles seen but not (yet) certified
float64 thrust                # commanded collective thrust this tick, N

# --- disturbance estimate (UDE) ---
geometry_msgs/Vector3 d_hat   # world-frame disturbance estimate, N; zero
                              # unless ude_enabled and the estimator is active

# --- goal / progress ---
geometry_msgs/Point waypoint  # active goal the core is tracking
bool reached                  # dist_to_goal < goal_arrival_tolerance
float64 dist_to_goal          # m
float64 min_obstacle_dist     # surface distance to the nearest obstacle in
                              # the buffer the QP used, m; large sentinel
                              # (1e9) when no obstacles are visible
