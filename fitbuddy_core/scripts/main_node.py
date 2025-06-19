#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from path_planner.path_planner import PathPlanner  # adjust the import if needed

class Fitbuddy:
    def __init__(self):
        rospy.init_node("main_node", anonymous=True)

        self.target_position = None
        self.obstacle_position = np.array([])

        rospy.Subscriber("/target_position", Point, self.target_callback)
        rospy.Subscriber("/lidar/obstacles", Marker, self.obstacle_callback)

        self.planner = PathPlanner()

        self.timer = rospy.Timer(rospy.Duration(0.5), self.timer_callback)

    def target_callback(self, msg):
        self.target_position = np.array([msg.x, msg.y])
        rospy.loginfo(f"Updated Target Position: {self.target_position}")

    def obstacle_callback(self, marker_msg):
        self.obstacle_position = np.array([[p.x, p.y] for p in marker_msg.points])
        rospy.loginfo(f"Updated Obstacle Positions: {self.obstacle_position.shape[0]} obstacles")

    def timer_callback(self, event):
        if self.target_position is None or self.obstacle_position.size == 0:
            rospy.logwarn("[Fitbuddy] Waiting for both target and obstacles data...")
            return

        try:
            control_input = self.planner.update(self.target_position, self.obstacle_position)
            rospy.loginfo(f"Control Input: {control_input}")
        except Exception as e:
            rospy.logerr(f"Planner update failed: {e}")

if __name__ == '__main__':
    try:
        node = Fitbuddy()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
