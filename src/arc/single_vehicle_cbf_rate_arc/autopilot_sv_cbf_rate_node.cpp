#include "rclcpp/rclcpp.hpp"
#include "single_vehicle_cbf_rate/single_vehicle_cbf_rate_client.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<nodelib::RateAutopilotClient>(rclcpp::NodeOptions()));
  rclcpp::shutdown();
  return 0;
}
