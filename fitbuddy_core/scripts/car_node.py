#!/usr/bin/env python3

import numpy as np
import serial
import rospy

class RobotCar:
    def __init__(self):
        self.car_serial = None

        try:
            self.car_serial = serial.Serial('/dev/robot_car', 115200, timeout=0.1)
        except Exception as e:
            rospy.logerr(f"[CarNode] Failed to connect to one of the serial ports: {e}")
            return

    def setPos(self, val):
        return
    
    def setVel(self, control_input):
        vx, vy = control_input

        v = vx
        omega = np.arctan2(vx, vy)

        A = 0.125 / 2 # Half of width
        B = 0.1 / 2 # Half of length
        
        # It is set to position direction
        v_left = -int(v - omega * (A + B) * 1000) # mm/s
        v_right = -int(v + omega * (A + B) * 1000) # mm/s

        self.car_serial.write(f"L:{v_left},R:{v_right}\n")



        # 2pi r / 52 encoder