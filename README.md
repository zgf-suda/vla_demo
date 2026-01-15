# vla_demo
任务：franka机器人桌面清理任务 
相机：1个在手腕，一个放桌上的固定位置
仿真中训练groot n1.5，数据是在仿真中遥操的，模型服务通过HTTP传给Isaacsim来仿真。

# 代码简介
keyboard_tele.py:键盘遥操
server_franka.py：模型服务
client_sim.py:sim仿真
