# ros2_drone_sim
修改后的仿真代码，测试最高可跑到89.2
## First
打开QGC
## Terminator 1-3
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


## Terminator 2-3
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
