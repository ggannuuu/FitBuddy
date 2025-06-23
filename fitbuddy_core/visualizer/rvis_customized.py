#!/usr/bin/env python3

import rospy
import subprocess
import os
import tempfile

def create_rviz_config():
    """Create a custom RViz configuration for visualizing the MPC system"""
    
    rviz_config = """
Panels:
  - Class: rviz/Displays
    Help Height: 78
    Name: Displays
    Property Tree Widget:
      Expanded:
        - /Global Options1
        - /Status1
        - /Obstacles1
        - /Target1
        - /Control Vector1
      Splitter Ratio: 0.5
    Tree Height: 549
  - Class: rviz/Selection
    Name: Selection
  - Class: rviz/Tool Properties
    Expanded:
      - /2D Pose Estimate1
      - /2D Nav Goal1
      - /Publish Point1
    Name: Tool Properties
    Splitter Ratio: 0.588235259056091
  - Class: rviz/Views
    Expanded:
      - /Current View1
    Name: Views
    Splitter Ratio: 0.5
  - Class: rviz/Time
    Experimental: false
    Name: Time
    SyncMode: 0
    SyncSource: ""
Preferences:
  PromptSaveOnExit: true
Toolbars:
  toolButtonStyle: 2
Visualization Manager:
  Class: ""
  Displays:
    - Alpha: 0.5
      Cell Size: 1
      Class: rviz/Grid
      Color: 160; 160; 164
      Enabled: true
      Line Style:
        Line Width: 0.029999999329447746
        Value: Lines
      Name: Grid
      Normal Cell Count: 0
      Offset:
        X: 0
        Y: 0
        Z: 0
      Plane: XY
      Plane Cell Count: 20
      Reference Frame: <Fixed Frame>
      Value: true
    - Class: rviz/MarkerArray
      Enabled: true
      Marker Topic: /visualization/obstacles
      Name: Obstacles
      Namespaces:
        obstacles: true
      Queue Size: 100
      Value: true
    - Class: rviz/Marker
      Enabled: true
      Marker Topic: /visualization/target
      Name: Target
      Namespaces:
        target: true
      Queue Size: 100
      Value: true
    - Class: rviz/Marker
      Enabled: true
      Marker Topic: /visualization/control_vector
      Name: Control Vector
      Namespaces:
        control_vector: true
      Queue Size: 100
      Value: true
    - Class: rviz/Marker
      Enabled: true
      Marker Topic: /lidar/obstacles
      Name: Lidar Obstacles
      Namespaces:
        {}
      Queue Size: 100
      Value: true
  Enabled: true
  Global Options:
    Background Color: 48; 48; 48
    Default Light: true
    Fixed Frame: map
    Frame Rate: 30
  Name: root
  Tools:
    - Class: rviz/Interact
      Hide Inactive Objects: true
    - Class: rviz/MoveCamera
    - Class: rviz/Select
    - Class: rviz/FocusCamera
    - Class: rviz/Measure
    - Class: rviz/SetInitialPose
      Theta std deviation: 0.2617993950843811
      Topic: /initialpose
      X std deviation: 0.5
      Y std deviation: 0.5
    - Class: rviz/SetGoal
      Topic: /move_base_simple/goal
    - Class: rviz/PublishPoint
      Single click: true
      Topic: /clicked_point
  Value: true
  Views:
    Current:
      Class: rviz/Orbit
      Distance: 10
      Enable Stereo Rendering:
        Stereo Eye Separation: 0.05999999865889549
        Stereo Focal Distance: 1
        Swap Stereo Eyes: false
        Value: false
      Field of View: 0.7853981852531433
      Focal Point:
        X: 0
        Y: 0
        Z: 0
      Focal Shape Fixed Size: true
      Focal Shape Size: 0.05000000074505806
      Invert Z Axis: false
      Name: Current View
      Near Clip Distance: 0.009999999776482582
      Pitch: 1.5697963237762451
      Target Frame: <Fixed Frame>
      Yaw: 0
    Saved: ~
Window Geometry:
  Displays:
    collapsed: false
  Height: 846
  Hide Left Dock: false
  Hide Right Dock: false
  QMainWindow State: 000000ff00000000fd000000040000000000000156000002b0fc0200000008fb0000001200530065006c0065006300740069006f006e00000001e10000009b0000005c00fffffffb0000001e0054006f006f006c002000500072006f007000650072007400690065007302000001ed000001df00000185000000a3fb000000120056006900650077007300200054006f006f02000001df000002110000018500000122fb000000200054006f006f006c002000500072006f0070006500720074006900650073003203000002880000011d000002210000017afb000000100044006900730070006c006100790073010000003d000002b0000000c900fffffffb0000002000730065006c0065006300740069006f006e00200062007500660066006500720200000138000000aa0000023a00000294fb00000014005700690064006500670065007400730000000000000000000000000000000000fb0000000c004b0069006e0065006300740200000186000001060000030c00000261000000010000010f000002b0fc0200000003fb0000001e0054006f006f006c002000500072006f00700065007200740069006500730100000041000000780000000000000000fb0000000a00560069006500770073010000003d000002b0000000a400fffffffb0000001200530065006c0065006300740069006f006e010000025a000000b200000000000000000000000200000490000000a9fc0100000001fb0000000a00560069006500770073030000004e00000080000002e10000019700000003000004420000003efc0100000002fb0000000800540069006d00650100000000000004420000000000000000fb0000000800540069006d00650100000000000004500000000000000000000002cc000002b000000004000000040000000800000008fc0000000100000002000000010000000a0054006f006f006c00730100000000ffffffff0000000000000000
  Selection:
    collapsed: false
  Time:
    collapsed: false
  Tool Properties:
    collapsed: false
  Views:
    collapsed: false
  Width: 1090
  X: 830
  Y: 234
"""
    return rviz_config

def launch_rviz():
    """Launch RViz with the custom configuration"""
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.rviz', delete=False) as f:
        f.write(create_rviz_config())
        config_file = f.name
    
    try:
        # Launch RViz with the config file
        rospy.loginfo(f"Launching RViz with config: {config_file}")
        subprocess.run(['rviz', '-d', config_file], check=True)
    except subprocess.CalledProcessError as e:
        rospy.logerr(f"Failed to launch RViz: {e}")
    except KeyboardInterrupt:
        rospy.loginfo("RViz launch interrupted")
    finally:
        # Clean up the temporary file
        if os.path.exists(config_file):
            os.unlink(config_file)

def main():
    """Main function to initialize and launch visualization"""
    rospy.init_node('rviz_launcher', anonymous=True)
    rospy.loginfo("Starting custom RViz visualization for FitBuddy MPC")
    
    try:
        launch_rviz()
    except rospy.ROSInterruptException:
        rospy.loginfo("RViz launcher shutting down")

if __name__ == '__main__':
    main()