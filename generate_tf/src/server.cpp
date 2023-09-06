#include "custom_interfaces/srv/tf.hpp"
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_ros/transform_broadcaster.h"

class TfService : public rclcpp::Node
{
public:
    TfService() : Node("Tf_server_node")
    {
        srv_ = this->create_service<custom_interfaces::srv::Tf>("get_tf_server", 
            std::bind(&TfService::serverCallback, this, std::placeholders::_1, std::placeholders::_2));
        
        broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100), std::bind(&TfService::publishTf, this));
    }

private:
    void publishTf()
    {
        if(pub_tf_)
        {
            broadcaster_->sendTransform(t1_);
            broadcaster_->sendTransform(t2_);
        }
    }

    void serverCallback(
        const std::shared_ptr<custom_interfaces::srv::Tf::Request> request,
        std::shared_ptr<custom_interfaces::srv::Tf::Response> response)
    {
        t1_.header.stamp = this->now();
        t1_.header.frame_id = "camera_link";
        t1_.child_frame_id = "objeto_link";
        t1_.transform.translation.x = request->x / 1000.0;
        t1_.transform.translation.y = request->y / 1000.0;
        t1_.transform.translation.z = request->z / 1000.0;
        t1_.transform.rotation.w = 1.0;
        broadcaster_->sendTransform(t1_);

        t2_.header.stamp = this->now();
        t2_.header.frame_id = "objeto_link";
        t2_.child_frame_id = "objeto_goal";
        t2_.transform.translation.x = request->x / 1000.0 + 0.07;
        t2_.transform.translation.y = request->y / 1000.0;
        t2_.transform.translation.z = request->z / 1000.0;
        t2_.transform.rotation.w = 1.0;
        broadcaster_->sendTransform(t2_);

        response->succeed = true;
    }

    rclcpp::Service<custom_interfaces::srv::Tf>::SharedPtr srv_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> broadcaster_;
    rclcpp::TimerBase::SharedPtr timer_;
    geometry_msgs::msg::TransformStamped t1_, t2_;
    bool pub_tf_ = false;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto service = std::make_shared<TfService>();
    rclcpp::spin(service);
    rclcpp::shutdown();
    return 0;
}