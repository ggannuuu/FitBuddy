#!/usr/bin/env python3

import rospy
import serial
from std_msgs.msg import String


class UWBPublisher:
    def __init__(self):
        rospy.init_node('uwb_node')

        self.anchor_port = {}
        self.pub = rospy.Publisher('/uwb_distance', String, queue_size=10)

        try:
            self.anchor_port = {
                "A1": serial.Serial('/dev/ttyUSB1', 115200, timeout=0.1),
                "A2": serial.Serial('/dev/ttyUSB2', 115200, timeout=0.1),
                "A3": serial.Serial('/dev/ttyUSB3', 115200, timeout=0.1)
            }
        except Exception as e:
            rospy.logerr(f"[UWBNode] Failed to connect to one of the serial ports: {e}")
            return

        rospy.Timer(rospy.Duration(1.0 / 4), self.read_from_anchor)

    def read_from_anchor(self, event):
        for anchor_name, port in self.anchor_port.items():
            try:
                line = port.readline().decode('utf-8').strip()
                if line:
                    msg = f"{anchor_name} {line}"
                    rospy.loginfo(msg)
                    self.pub.publish(msg)
            except Exception as e:
                rospy.logerr(f"[UWBNode] Error reading from {anchor_name}: {e}")

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = UWBPublisher()
        node.run()
    except rospy.ROSInterruptException:
        pass
