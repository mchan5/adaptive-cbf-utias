// Tier 0 unit test for the selectable obstacle model in
// body_rate_hocbf.cpp::collisionBarrierChain. See
// PLAN_cylinder_barrier_20260831.md and penn_gat_training/TESTING.md.
//
// collisionBarrierChain's trailing `cylinder_mode` argument switches the
// obstacle geometry:
//   false (default) -- sphere: full 3-D distance; a vertical-only offset is a
//     real constraint (this is the original certified behavior).
//   true -- vertical cylinder: the barrier and its whole relative-degree-2
//     chain depend only on the HORIZONTAL relative position, so an obstacle
//     directly above or below the vehicle is completely ignored.
#include <gtest/gtest.h>

#include "single_vehicle_cbf_rate/body_rate_hocbf.hpp"

using nodelib::BarrierChain;
using nodelib::QuadState;
using nodelib::collisionBarrierChain;

namespace {

constexpr double kDMin = 1.0;
constexpr double kG1 = 4.0;
constexpr double kG2 = 4.0;
constexpr double kMass = 2.064;

QuadState hoverState() {
  QuadState s;
  s.p = Eigen::Vector3d(0.0, 0.0, 1.5);
  s.v = Eigen::Vector3d(0.5, 1.0, 0.3);  // deliberately has a vertical component
  s.R = Eigen::Matrix3d::Identity();
  s.T = kMass * nodelib::kQuadG;
  return s;
}

}  // namespace

// Cylinder mode: an obstacle directly below (or above) the vehicle -- purely
// vertical offset -- must produce h = -d_min^2 and identically zero control
// coefficients: the QP sees no constraint from it at all.
TEST(CylinderBarrier, PureVerticalOffsetIsIgnored) {
  const QuadState s = hoverState();

  for (double dz : {-3.0, -0.5, 0.5, 3.0}) {
    const Eigen::Vector3d p_obs = s.p - Eigen::Vector3d(0.0, 0.0, dz);
    const BarrierChain c =
        collisionBarrierChain(s, p_obs, kDMin, kG1, kG2, kMass, /*cylinder_mode=*/true);

    EXPECT_NEAR(c.h, -kDMin * kDMin, 1e-12) << "dz=" << dz;
    EXPECT_NEAR(c.psi0, -kDMin * kDMin, 1e-12) << "dz=" << dz;
    EXPECT_NEAR(c.c_tau, 0.0, 1e-12) << "dz=" << dz;
    EXPECT_NEAR(c.c_omega1, 0.0, 1e-12) << "dz=" << dz;
    EXPECT_NEAR(c.c_omega2, 0.0, 1e-12) << "dz=" << dz;
    EXPECT_NEAR(c.r_tilde.norm(), 0.0, 1e-12) << "dz=" << dz;
  }
}

// Cylinder mode: h is the squared horizontal centre distance minus d_min^2.
TEST(CylinderBarrier, HorizontalDistanceDefinesH) {
  const QuadState s = hoverState();
  const Eigen::Vector3d p_obs = s.p - Eigen::Vector3d(1.2, -0.9, 0.0);
  const BarrierChain c =
      collisionBarrierChain(s, p_obs, kDMin, kG1, kG2, kMass, /*cylinder_mode=*/true);

  const double expected_h = 1.2 * 1.2 + 0.9 * 0.9 - kDMin * kDMin;
  EXPECT_NEAR(c.h, expected_h, 1e-12);
}

// Cylinder mode: sliding the obstacle purely in z leaves the entire chain
// unchanged -- altitude does not matter to a vertical cylinder.
TEST(CylinderBarrier, AltitudeOffsetDoesNotChangeChain) {
  const QuadState s = hoverState();
  const Eigen::Vector3d horiz(1.2, -0.9, 0.0);

  const BarrierChain base = collisionBarrierChain(s, s.p - horiz, kDMin, kG1, kG2, kMass,
                                                  /*cylinder_mode=*/true);
  const BarrierChain lifted =
      collisionBarrierChain(s, s.p - horiz - Eigen::Vector3d(0.0, 0.0, 2.5), kDMin, kG1, kG2,
                            kMass, /*cylinder_mode=*/true);

  EXPECT_NEAR(base.h, lifted.h, 1e-12);
  EXPECT_NEAR(base.psi1, lifted.psi1, 1e-12);
  EXPECT_NEAR(base.psi2, lifted.psi2, 1e-12);
  EXPECT_NEAR(base.L_ftilde, lifted.L_ftilde, 1e-12);
  EXPECT_NEAR(base.c_tau, lifted.c_tau, 1e-12);
  EXPECT_NEAR(base.c_omega1, lifted.c_omega1, 1e-12);
  EXPECT_NEAR(base.c_omega2, lifted.c_omega2, 1e-12);
}

// Sphere mode is the default and is unchanged by this feature: a purely
// vertical offset IS a real constraint, and the chain still depends on z.
TEST(SphereBarrier, DefaultStillSeesVerticalObstacle) {
  const QuadState s = hoverState();
  const Eigen::Vector3d p_obs = s.p - Eigen::Vector3d(0.0, 0.0, 0.5);

  const BarrierChain def = collisionBarrierChain(s, p_obs, kDMin, kG1, kG2, kMass);
  const BarrierChain explicit_sphere =
      collisionBarrierChain(s, p_obs, kDMin, kG1, kG2, kMass, /*cylinder_mode=*/false);

  // default == explicit sphere
  EXPECT_NEAR(def.h, explicit_sphere.h, 1e-12);
  EXPECT_NEAR(def.c_tau, explicit_sphere.c_tau, 1e-12);

  // and it is a genuine constraint, unlike cylinder mode
  const double expected_h = 0.5 * 0.5 - kDMin * kDMin;
  EXPECT_NEAR(def.h, expected_h, 1e-12);
  EXPECT_GT(std::abs(def.c_tau), 1e-6);
}
