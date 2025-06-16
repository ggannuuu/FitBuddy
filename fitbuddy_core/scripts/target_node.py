#!/usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import Point
from scipy.optimize import minimize, NonlinearConstraint

class TargetEstimator:
    def __init__(self):
        self.anchors = {}
        self.anchor_centers = [
            np.array([0,  np.sqrt(3)/2, 0]),
            np.array([-1, -np.sqrt(3)/2, 0]),
            np.array([1,  -np.sqrt(3)/2, 0])   
        ]
        self.pub = rospy.Publisher('/target_position', Point, queue_size=10)
        rospy.Subscriber('/uwb_distance', String, self.uwb_callback)

    def uwb_callback(self, msg):
        try:
            anchor_id, distance = msg.data.split()
            self.anchors[anchor_id] = float(distance)

            if len(self.anchors) == 3:
                pos = self.estimate_target()
                point_msg = Point(x=pos[0], y=pos[1], z=pos[2])
                self.pub.publish(point_msg)
                rospy.loginfo(f"Target position: {pos}")
        except Exception as e:
            rospy.logwarn(f"[TargetEstimator] Failed to parse msg: {msg.data}, error: {e}")

    def estimate_target(self):
        dists = [self.anchors["A1"], self.anchors["A2"], self.anchors["A3"]]
        c1, c2, c3 = self.anchor_centers

        def objective(vars):
            x = vars[0:3]
            s1, s2, s3 = vars[3:6], vars[6:9], vars[9:12]
            return np.linalg.norm(x - s1) + np.linalg.norm(x - s2) + np.linalg.norm(x - s3)

        constraints = [
            NonlinearConstraint(lambda v: np.linalg.norm(v[3:6] - c1) - dists[0], 0, 0),
            NonlinearConstraint(lambda v: np.linalg.norm(v[6:9] - c2) - dists[1], 0, 0),
            NonlinearConstraint(lambda v: np.linalg.norm(v[9:12] - c3) - dists[2], 0, 0),
        ]

        x0 = np.concatenate([
            np.mean(self.anchor_centers, axis=0),
            c1 + [dists[0], 0, 0],
            c2 + [dists[1], 0, 0],
            c3 + [dists[2], 0, 0],
        ])

        result = minimize(objective, x0, constraints=constraints, method="SLSQP")
        return result.x[:3]


if __name__ == "__main__":
    rospy.init_node("target_node")
    estimator = TargetEstimator()
    rospy.spin()
