#RoboEyes

A face recognition project for ROS.

## USAGE

把core后台跑起来


```sh
nohup roscore &
cd ~/catkin_ws/src #定位到自己的workspace
git clone https://github.com/shleeky/robo_eyes.git #clone源码到本地
cd ~/catkin_ws
catkin_make
. ~/catkin_ws/devel/setup.bash
```

这个时候应该已经把robo_eyes包装起来了。
现在我们要把节点跑起来。

```sh
roscd robo_eyes
cd ./scripts
rosrun robo_eyes cloud_server.py
```

再打开另外一个端口，同样定位到robo_eyes目录下，运行web_cam_ser:

```sh
rosrun robo_eyes webcam_ser.py
```

