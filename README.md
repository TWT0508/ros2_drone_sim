# tep-run
## First
打开QGC
## terminator 1-3
```bash
# 进入项目目录
cd /home/w/Desktop/ros2_drone_sim
python3 offboard_ros2.py
```

```bash
# 进入项目目录
cd /home/w/Desktop/ros2_drone_sim 
python3 visualization_scoring.py 
```

```bash
rviz2
```


## terminator 2-3
```bash
source /opt/ros/humble/setup.bash
cd ~/PX4-Autopilot/
make px4_sitl gz_x500
```

```bash
ros2 topic list
```

```bash
ros2 launch mavros px4.launch fcu_url:=udp://:14540@127.0.0.1:14580
```
