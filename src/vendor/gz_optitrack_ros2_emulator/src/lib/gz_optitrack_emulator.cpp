/*
MIT License

Copyright (c) 2025 FSC Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "gz_optitrack_emulator.hpp"
#include <functional>
#include "math_utils.hpp"


GazeboMocapEmulator::GazeboMocapEmulator(const rclcpp::NodeOptions & options):
    Node("gazebo_mocap_emulator", options)
{
    LoadParameters();
    ConstructTopics();
    SetupNode();
}

void GazeboMocapEmulator::LoadParameters()
{
    this->declare_parameter<ModeList>(modelListName, ModeList());
    // determine whether parameter file is valid
    if(!this->get_parameter(modelListName, modelList)) {
        RCLCPP_ERROR(this->get_logger(), "Failed to load model list parameter.");
        rclcpp::shutdown();
        std::exit(1);  // or return from main with 1
    }
    if(modelList.size() == 0) {
        RCLCPP_ERROR(this->get_logger(), "There is no valid model is the parameter file! Check the name of the param.");
        rclcpp::shutdown();
        std::exit(1);  // or return from main with 1       
    }
    RCLCPP_INFO(this->get_logger(), "The following models are detected:");
    for (const auto& s : modelList) {
        RCLCPP_INFO(this->get_logger(), "%s", s.c_str());
    }
}

void GazeboMocapEmulator::ConstructTopics()
{
    for (const auto& s : modelList) {
        topicPair.emplace_back("/model/" + s + "/groundtruth_odometry",
            "/" + s + "/mocap");
        RCLCPP_INFO(this->get_logger(),
            "subscribe to gz topic: %s. publish to ros topic: %s",
            topicPair.back().first.c_str(),
            topicPair.back().second.c_str());
    }
}

void GazeboMocapEmulator::SetupNode()
{
    for (size_t i = 0; i < modelList.size(); i++) {
        publisherMap[modelList[i]] = this->create_publisher<MocapType>(topicPair[i].second, qos);
        const std::string model_name = modelList[i];
        const std::string gz_topic = topicPair[i].first;
        // pass callable object ot the subscriber to create multiple subscriber topics
        std::function<void(const GzMsgType &)> callback =
            [this, model_name](const GzMsgType &msg) {
                this->GzCallBack(model_name, msg);
            };
        if (!gzNode.Subscribe(gz_topic, callback)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to subscribe to: %s", gz_topic.c_str());
            // if subscribe can not be built, exit
            rclcpp::shutdown();
            std::exit(1);  // or return from main with 1       
        }
    }
}

void GazeboMocapEmulator::GzCallBack(const std::string &model, const GzMsgType &msg)
{
    MocapType mocapMsg;
    rclcpp::Clock sys_clock(RCL_SYSTEM_TIME);
    mocapMsg.header.stamp = sys_clock.now(); //
    mocapMsg.header.frame_id = "map";

    mocapMsg.pose.position.x = msg.pose().position().x();
    mocapMsg.pose.position.y = msg.pose().position().y();
    mocapMsg.pose.position.z = msg.pose().position().z();

    mocapMsg.pose.orientation.x = msg.pose().orientation().x();
    mocapMsg.pose.orientation.y = msg.pose().orientation().y();
    mocapMsg.pose.orientation.z = msg.pose().orientation().z();
    mocapMsg.pose.orientation.w = msg.pose().orientation().w();

    // After copying from Gazebo:
    auto &o = mocapMsg.pose.orientation;
    double n = std::sqrt(o.w*o.w + o.x*o.x + o.y*o.y + o.z*o.z);
    if (n > 1e-12) { o.w/=n; o.x/=n; o.y/=n; o.z/=n; }
    else { o.w = 1.0; o.x = o.y = o.z = 0.0; }  // guard against NaN
    // Then build q (no second normalize needed)

    Eigen::Quaterniond q_FLU_to_ENU(
        mocapMsg.pose.orientation.w,
        mocapMsg.pose.orientation.x,
        mocapMsg.pose.orientation.y,
        mocapMsg.pose.orientation.z);

    // Normalize just in case (optional)
    q_FLU_to_ENU.normalize();

    // Gazebo linear velocity is measured in FLU, we need to convert it to ENU
    const Eigen::Vector3d v_flu(msg.twist().linear().x(),
                                msg.twist().linear().y(),
                                msg.twist().linear().z());

    const Eigen::Vector3d v_enu = q_FLU_to_ENU * v_flu;

    mocapMsg.twist.linear.x = v_enu.x();
    mocapMsg.twist.linear.y = v_enu.y();
    mocapMsg.twist.linear.z = v_enu.z();

    // gazebo angular velocity is measrued in FLU already
    Eigen::Vector3d w_flu(
    msg.twist().angular().x(),
    msg.twist().angular().y(),
    msg.twist().angular().z());

    // Write back to your Mocap message
    mocapMsg.twist.angular.x = w_flu.x();
    mocapMsg.twist.angular.y = w_flu.y();
    mocapMsg.twist.angular.z = w_flu.z();    

    auto it = publisherMap.find(model);
    if (it != publisherMap.end()) {
      it->second->publish(mocapMsg);
    } else {
      RCLCPP_WARN(this->get_logger(), "No publisher found for model: %s", model.c_str());
    }
}
