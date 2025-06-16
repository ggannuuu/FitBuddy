# 🤖 Autonomous Following Bot

This repository contains the full-stack implementation of an **Autonomous Following Robot** that uses **Ultra-Wideband (UWB)** for target tracking, **LiDAR** for obstacle detection, and a hybrid **MPC + PCIP + L1 Adaptive Optimization** algorithm for motion planning and control. The robot operates on a **ROS** architecture and runs on a **Raspberry Pi-powered mobile platform**.

> 🧠 Built for dynamic indoor environments with real-time responsiveness and robust performance.

---

## 🗺️ Features

- 🚀 Real-time **target following** using UWB trilateration
- 🌳 **LiDAR-based** obstacle detection and avoidance
- 🔄 Robust trajectory tracking with:
  - **Model Predictive Control (MPC)**
  - **Prediction-Correction Interior-Point (PCIP)** solver
  - **L1 Adaptive Optimization (L1AO)** for uncertainty compensation
- 🚗 Modular software stack on **ROS**
- 🌟 Designed for **Raspberry Pi + robot car** platforms

---

## 🧠 Algorithm Stack

### Model Predictive Control (MPC)
Solves a finite-horizon optimal control problem in real-time, generating control inputs that guide the robot to follow the UWB target while respecting safety and actuator constraints.

### PCIP Solver
A fast, time-varying optimization method that accelerates convergence by applying prediction and correction steps, making it ideal for dynamic environments with changing constraints and goals.

### L1 Adaptive Optimization (L1AO)
Introduces robustness against modeling errors, sensor noise, and external disturbances. L1AO adjusts the optimization solution in real time, ensuring stability and consistent tracking performance.

---

## 📚 System Overview

### ROS (Robot Operating System)
- Integration of all components (UWB, LiDAR, planning, control)
- Real-time message passing and modularity
- Visualization through `rviz` and debugging with `rqt`

### UWB Target Localization
- 1 Tag (on person or target)
- 3 Anchors (fixed in environment)
- Uses **DW3000** modules for accurate trilateration

### LiDAR Perception
- 2D scanning with **RPLiDAR A1/A2**
- Builds occupancy grid
- Detects dynamic and static obstacles for avoidance

### Raspberry Pi + Robot Car
- ROS nodes run on **Raspberry Pi 4/5**
- Controls DC motors via PWM or motor drivers
- Chassis can be differential drive or Ackermann steering


---

## 🚧 Installation

### Requirements
- ROS Noetic
- Python 3.8+
- LiDAR SDK (e.g., RPLiDAR SDK)
- `numpy`, `scipy`, `matplotlib`, `pyserial`, `pylidar`
- Custom solvers for PCIP and L1AO

### Build & Run
```bash
git clone https://github.com/ggannuuu/FitBuddy.git
cd fitbuddy_ws
catkin_make
source devel/setup.bash
roslaunch launch/following_bot.launch
```

---

## 🎥 Demo

Comming Soon!

---

## 📚 References

- Hovakimyan et al., "L1 Adaptive Optimization for Autonomous Systems", *2023*
- Kim et al., "Real-Time MPC for Mobile Robots with Dynamic Obstacles", *2024*
- ROS, RPLiDAR SDK, Qorvo DW3000 UWB Modules

---

## 🧑‍💻 Author

**Geonwoo Kim**  
[GitHub](https://github.com/ggannuuu)
