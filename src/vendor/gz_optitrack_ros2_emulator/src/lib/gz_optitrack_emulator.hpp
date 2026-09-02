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

#ifndef GZ_OPTITRACK_EMULATOR_HPP
#define GZ_OPTITRACK_EMULATOR_HPP

#include <iostream>
#include <array>
#include <utility>

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <gz/transport/Node.hh>
#include <gz/msgs/clock.pb.h>
#include <gz/msgs/odometry.pb.h> // gz::msgs::Odometry

#include <fsc_autopilot_ros2_msgs/msg/mocap.hpp>

class GazeboMocapEmulator : public rclcpp::Node
{
public:
    explicit GazeboMocapEmulator(const rclcpp::NodeOptions & options);
    using ModeList = std::vector<std::string>;
    using TopicPair = std::pair<std::string, std::string>;
    // used to construct a publisher map
    using MocapType = fsc_autopilot_ros2_msgs::msg::Mocap;
    using MocapPub = rclcpp::Publisher<MocapType>::SharedPtr;
    using GzMsgType = gz::msgs::Odometry;
private:
    static constexpr const char* modelListName = "gz_model_list";
    static constexpr uint32_t qos = 10u; // quality of service in the publisher and subscriber
    // load model names
    void LoadParameters();
    // construct the gz_topic_name and ros_topic_name according to the model name
    void ConstructTopics();
    void SetupNode();
    void GzCallBack(const std::string &model, const GzMsgType &msg);
    ModeList modelList;
    std::vector<TopicPair> topicPair;
    gz::transport::Node gzNode;
    std::map<std::string, MocapPub> publisherMap;
};

#endif
