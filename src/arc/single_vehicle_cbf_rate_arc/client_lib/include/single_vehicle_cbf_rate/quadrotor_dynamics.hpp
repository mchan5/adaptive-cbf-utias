#pragma once

#include <Eigen/Dense>

namespace nodelib {

inline constexpr double kQuadG = 9.81;

struct QuadState {
  Eigen::Vector3d p{Eigen::Vector3d::Zero()};
  Eigen::Vector3d v{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d R{Eigen::Matrix3d::Identity()};
  double T{0.0};
};

}  