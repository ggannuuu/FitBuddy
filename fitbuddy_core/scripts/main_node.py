#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import Point, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from path_planner.l1ao_mpc_planner import PathPlanner 

class Fitbuddy:
    def __init__(self):
        rospy.init_node("main_node", anonymous=True)
        
        # Initialize instance variables
        self.target_position = None
        self.obstacle_position = []
        self.obstacle_radius = 0.4
        self.target_radius = 0.5
        self.control_input = np.array([0.0, 0.0])
        
        # Initialize publishers for visualization
        self.obstacles_pub = rospy.Publisher('/visualization/obstacles', MarkerArray, queue_size=10)
        self.target_pub = rospy.Publisher('/visualization/target', Marker, queue_size=10)
        self.control_vector_pub = rospy.Publisher('/visualization/control_vector', Marker, queue_size=10)

        rospy.sleep(2)

        rospy.Subscriber("/target_position", Point, self.target_callback)
        target_msg = rospy.wait_for_message("/target_position", Point)
        target_init = np.array([target_msg.x, target_msg.y])

        rospy.sleep(2)


        rospy.Subscriber("/lidar/obstacles", Marker, self.obstacle_callback)
        marker_msg = rospy.wait_for_message("/lidar/obstacles", Marker)
        
        # Initialize with fixed number of obstacles
        FIXED_OBSTACLE_COUNT = 15
        if marker_msg.points:
            # Process the initial message through obstacle_callback logic
            actual_obstacles = [
                (np.array([p.x, p.y]), self.obstacle_radius)
                for p in marker_msg.points
            ]
            if len(actual_obstacles) >= FIXED_OBSTACLE_COUNT:
                obstacle_init = actual_obstacles[:FIXED_OBSTACLE_COUNT]
            else:
                obstacle_init = actual_obstacles[:]
                while len(obstacle_init) < FIXED_OBSTACLE_COUNT:
                    obstacle_init.append((np.array([100.0, 100.0]), self.obstacle_radius))
            rospy.loginfo(f"Obstacle init: {len(obstacle_init)} obstacles")
        else:
            # Create dummy obstacles for initialization
            obstacle_init = [(np.array([100.0, 100.0]), self.obstacle_radius) for _ in range(FIXED_OBSTACLE_COUNT)]
            rospy.loginfo("No inputs from lidar - using dummy obstacles")

        self.planner = PathPlanner(
            horizon = 5,
            dt = 0.5,
            Q_f = np.eye(2) * 10.0,
            R = np.eye(2) * 1.0,
            obstacle_init = obstacle_init,
            obstacle_radius = 0.4,
            target_init = target_init,
            dt_mpc = 0.5,
            robot_radius_val = 0.4,
            cost_param_c_val = 1.0)

        self.timer = rospy.Timer(rospy.Duration(0.5), self.timer_callback)

    def target_callback(self, msg):
        self.target_position = np.array([msg.x, msg.y])
        rospy.loginfo(f"Updated Target Position: {self.target_position}")

    def obstacle_callback(self, marker_msg):
        # Fixed number of obstacles (adjust this number as needed)
        FIXED_OBSTACLE_COUNT = 15
        
        # Get actual obstacles from lidar
        actual_obstacles = [
            (np.array([p.x, p.y]), self.obstacle_radius)
            for p in marker_msg.points
        ]
        
        if len(actual_obstacles) >= FIXED_OBSTACLE_COUNT:
            # Take first N obstacles if we have too many
            self.obstacle_position = actual_obstacles[:FIXED_OBSTACLE_COUNT]
        else:
            # Pad with dummy obstacles if we have too few
            self.obstacle_position = actual_obstacles[:]
            # Add dummy obstacles far away (e.g., at 100, 100)
            while len(self.obstacle_position) < FIXED_OBSTACLE_COUNT:
                self.obstacle_position.append((np.array([100.0, 100.0]), self.obstacle_radius))
        
        # rospy.loginfo(f"Updated Obstacle Positions: {len(self.obstacle_position)} obstacles")

    def timer_callback(self, event):
        if self.target_position is None or len(self.obstacle_position) == 0:
            rospy.logwarn("[Fitbuddy] Waiting for both target and obstacles data...")
            return

        try:
            control_input = self.planner.update(self.obstacle_position, self.target_position)
            self.control_input = control_input
            rospy.loginfo(f"Control Input: {control_input}")
            
            # Publish visualizations
            self.publish_visualizations()
            
        except Exception as e:
            rospy.logerr(f"Planner update failed: {e}")

    def publish_visualizations(self):
        """Publish visualization markers for obstacles, target, and control vector"""
        # Publish obstacles
        self.publish_obstacles()
        
        # Publish target
        self.publish_target()
        
        # Publish control vector
        self.publish_control_vector()
    
    def publish_obstacles(self):
        """Publish obstacle markers"""
        marker_array = MarkerArray()
        
        for i, (position, radius) in enumerate(self.obstacle_position):
            # Skip dummy obstacles far away
            if np.linalg.norm(position) > 50:
                continue
                
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "obstacles"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            # Position
            marker.pose.position.x = position[0]
            marker.pose.position.y = position[1]
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            
            # Scale
            marker.scale.x = radius * 2  # diameter
            marker.scale.y = radius * 2  # diameter
            marker.scale.z = 0.5  # height
            
            # Color (red for obstacles)
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.7
            
            marker_array.markers.append(marker)
        
        self.obstacles_pub.publish(marker_array)
    
    def publish_target(self):
        """Publish target marker"""
        if self.target_position is None:
            return
            
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "target"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        # Position
        marker.pose.position.x = self.target_position[0]
        marker.pose.position.y = self.target_position[1]
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        
        # Scale
        marker.scale.x = self.target_radius * 2  # diameter
        marker.scale.y = self.target_radius * 2  # diameter
        marker.scale.z = 0.5  # height
        
        # Color (green for target)
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        
        self.target_pub.publish(marker)
    
    def publish_control_vector(self):
        """Publish control input as an arrow vector"""
        if np.linalg.norm(self.control_input) < 0.01:  # Skip very small vectors
            return
            
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "control_vector"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        # Start point (robot at origin)
        start_point = Point()
        start_point.x = 0.0
        start_point.y = 0.0
        start_point.z = 0.1
        
        # End point (control input direction, scaled for visibility)
        scale_factor = 2.0  # Scale the arrow for better visibility
        end_point = Point()
        end_point.x = self.control_input[0] * scale_factor
        end_point.y = self.control_input[1] * scale_factor
        end_point.z = 0.1
        
        marker.points = [start_point, end_point]
        
        # Scale (arrow dimensions)
        marker.scale.x = 0.05  # shaft diameter
        marker.scale.y = 0.1   # head diameter
        marker.scale.z = 0.1   # head length
        
        # Color (blue for control vector)
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        
        self.control_vector_pub.publish(marker)

if __name__ == '__main__':
    try:
        node = Fitbuddy()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
