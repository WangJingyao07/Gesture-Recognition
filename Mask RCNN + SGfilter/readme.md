## How to use
This project is based on Mask RCNN and is written with optimizations such as SG filtering, which can realize real-time gesture key point detection, and further classes can be adjusted according to the set gesture posture.

run the "handPoseVideo.py" with camera;

or change the path inside the code for detecting images or videos

## 如何使用
该项目是在Mask RCNN的基础上引入SG滤波等优化编写而成，可以实现实时的手势关键点检测，同时根据所设定的手势姿态可进行进一步的classes调整。

运行handPoseVideo.py文件即可，可以在命令行输入-source 0即可调用摄像头，如果无法调用，则更改摄像头串口重新尝试（针对部分开发板分配摄像头串口非一般情况）

如果要检测某个路径下的图片或视频，则更改该python文件中的路径即可。
