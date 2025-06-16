#!/usr/bin/env python3

import rospy
import math
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PointStamped, Point
from std_msgs.msg import Header
from visualization_msgs.msg import Marker

def adaptive_sample(ranges, angle_min, angle_increment, range_min, range_max, delta_thresh=0.2):
    sampled = []
    angle = angle_min
    prev_valid = None

    for r in ranges:
        if math.isnan(r) or r < range_min or r > range_max:
            angle += angle_increment
            continue
        if prev_valid is None or abs(r - prev_valid) > delta_thresh:
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            sampled.append((x, y))
            prev_valid = r
        angle += angle_increment

    return sampled

class LidarNode:
    def __init__(self):
        rospy.init_node("lidar_node", anonymous=True)

        self.delta_thresh = rospy.get_param("~delta_thresh", 0.2)

        self.obstacle_pub = rospy.Publisher("/lidar/obstacles", Marker, queue_size=10)
        self.sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback)

    def scan_callback(self, data):
        sampled = adaptive_sample(
            data.ranges,
            data.angle_min,
            data.angle_increment,
            data.range_min,
            data.range_max,
            self.delta_thresh
        )

        marker = Marker()
        marker.header = Header()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "laser"  # or "base_link" depending on TF
        marker.ns = "lidar"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.2
        marker.color.b = 0.2

        for x, y in sampled:
            pt = Point()
            pt.x = x
            pt.y = y
            pt.z = 0.0
            marker.points.append(pt)

        self.obstacle_pub.publish(marker)

if __name__ == "__main__":
    try:
        node = LidarNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


# import rospy
# import serial
# import numpy as np
# import math
# from std_msgs.msg import String
# from scipy.optimize import minimize, NonlinearConstraint
# from sensor_msgs.msg import LaserScan


# def adaptive_sample(ranges, angle_min, angle_increment, range_min, range_max, delta_threash=0.2):
#     sampled = []
#     angle = angle_min
#     prev_valid = None

#     for r in ranges:
#         if math.isnan(r) or r < range_min or r > range_max:
#             angle += angle_increment
#             continue
#         if prev_valid is None or abs(r - prev_valid) > delta_threash:
#             x = r * np.cos(angle)
#             y = r * np.sin(angle)
#             sampled.append([x, y])
#             prev_valid = r        
        
#         angle += angle_increment

#     return sampled

# class Lidar():
#     def  __init__(self,
#                   delta_thresh = 0.2):
#         self.delta_thresh = delta_thresh
#         self.obstacle_coordinates = []
#         rospy.Subscriber('/scan', LaserScan, self.callback)

#     def callback(self, data):
#         self.obstacle_coordinates = adaptive_sample(data.ranges,
#                                                     data.angle_min,
#                                                     data.angle_increment,
#                                                     data.range_min,
#                                                     data.range_max,
#                                                     self.delta_thresh)
#         # rospy.loginfo(self.obstacle_coordinates)

#     def get_obstacle_coordinates(self):
#         return self.obstacle_coordinates