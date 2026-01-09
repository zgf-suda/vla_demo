import omni
from isaacsim import SimulationApp

config = {"headless": False, "physics_substeps": 16, "rendering_frequency": 60}
simulation_app = SimulationApp(config)

import numpy as np
import os
import cv2  # 确保安装了 opencv-python
from isaacsim.core.api import World
from isaacsim.core.prims import RigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api.robots import Robot
import omni.replicator.core as rep
from isaacsim.sensors.camera import CameraView
from isaacsim.core.api.articulations import ArticulationSubset
from isaacsim.core.utils.types import ArticulationAction
import sys

# 指向你的源码目录
gr00t_path = r"C:\Users\xjy47\Desktop\SIM\Isaac-GR00T-main"
if gr00t_path not in sys.path:
    sys.path.append(gr00t_path)
from gr00t.eval.robot import RobotInferenceClient

SERVER_HOST = "localhost"
SERVER_PORT = 5555
TASK_DESC = "put both the orange juice and the ketchup in the basket"


# === 预处理函数 (Client端完成缩放) ===
def preprocess_image(img_rgb):
    h, w, _ = img_rgb.shape  # 256, 256, 3
    # Center Crop & Resize to 224
    # 简单粗暴：直接Resize，为了先跑通
    img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    return img_resized


# === 场景搭建 (省略重复代码，保持你原有的) ===
my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()

# # 添加桌子

table_asset_path = r'C:\Users\xjy47\Desktop\SIM\table_clean/living_room_table/living_room_table.usd'
add_reference_to_stage(usd_path=table_asset_path, prim_path="/World/Table")
table = RigidPrim(
    "/World/Table", name="lab_table", positions=np.array([[0, 0, 0]]), scales=np.array([[1.5, 1.5, 1.5]]))


# 添加alphabet-soup

hole_asset_path = r'C:\Users\xjy47\Desktop\SIM\table_clean/alphabet_soup/alphabet_soup.usd'

add_reference_to_stage(usd_path=hole_asset_path, prim_path="/World/alphabet_soup")
my_world.scene.add(
    RigidPrim(
        "/World/alphabet_soup", name="alphabet_soup", positions=np.array([[-0.13, 0.22, 0.8]]),
        scales=np.array([[0.01, 0.01, 0.01]])
    ))

# basket
peg_asset_path = r'C:\Users\xjy47\Desktop\SIM\table_clean\basket\basket.usd'
add_reference_to_stage(usd_path=peg_asset_path, prim_path="/World/basket")
my_world.scene.add(
    RigidPrim(
        "/World/basket", name="basket", positions=np.array([[-0.4, 0.0, 0.8]]), scales=np.array([[1.5, 1.5, 1.5]])
    ))
# butter
hole_asset_path = r'C:\Users\xjy47\Desktop\SIM\table_clean\butter\butter.usd'
add_reference_to_stage(usd_path=hole_asset_path, prim_path="/World/butter")
my_world.scene.add(
    RigidPrim(
        "/World/butter", name="butter",  scales=np.array([[0.01, 0.01, 0.01]]),positions=np.array([[0.3, 0.0, 0.8]]),
            orientations=np.array([[0.3, 0, 0, 0.3]]),
    ))
# cream_cheese
hole_asset_path = os.path.abspath('./table_clean/cream_cheese/cream_cheese.usd')
add_reference_to_stage(usd_path=hole_asset_path, prim_path="/World/cream_cheese")
my_world.scene.add(
    RigidPrim(
        "/World/cream_cheese", name="cream_cheese", positions=np.array([[0.5, 0.0, 0.8]]),
        scales=np.array([[0.01, 0.01, 0.01]])
    ))
# ketchup
hole_asset_path = os.path.abspath('./table_clean/ketchup/ketchup.usd')
add_reference_to_stage(usd_path=hole_asset_path, prim_path="/World/ketchup")
my_world.scene.add(
    RigidPrim(
        "/World/ketchup", name="ketchup",scales=np.array([[0.01, 0.01, 0.01]]), positions=np.array([[-0.1, 0.0, 0.8]]),
    ))
# milk
hole_asset_path = os.path.abspath('./table_clean/milk/milk.usd')
add_reference_to_stage(usd_path=hole_asset_path, prim_path="/World/milk")
my_world.scene.add(
    RigidPrim(
        "/World/milk", name="milk", positions=np.array([[0.3, 0.3, 0.8]]), scales=np.array([[0.01, 0.01, 0.01]])
    ))
# orange_juice
hole_asset_path = os.path.abspath('./table_clean/orange_juice/orange_juice.usd')
add_reference_to_stage(usd_path=hole_asset_path, prim_path="/World/orange_juice")
my_world.scene.add(
    RigidPrim(
        "/World/orange_juice", name="orange_juice", positions=np.array([[0.1, 0.3, 0.8]]),
        scales=np.array([[0.01, 0.01, 0.01]])
    ))
# tomato_sauce
hole_asset_path = os.path.abspath('./table_clean/tomato_sauce/tomato_sauce.usd')
add_reference_to_stage(usd_path=hole_asset_path, prim_path="/World/tomato_sauce")
my_world.scene.add(
    RigidPrim(
        "/World/tomato_sauce", name="tomato_sauce",positions=np.array([[0.1, 0.0, 0.8]]),
        scales=np.array([[0.01, 0.01, 0.01]])
    ))
print("成功：场景中的桌子和物体均已添加。")

robot_usd_path = r'C:\Users\xjy47\Desktop\SIM\FrankaEmika\panda_instanceable.usd'
add_reference_to_stage(usd_path=robot_usd_path, prim_path="/World/Franka")
robot = my_world.scene.add(
    Robot("/World/Franka", name="franka_robot", position=np.array([[0.0, -0.3, 0.44]]),
          orientation=np.array([0.7071, 0, 0, 0.7071]))
)

KP_GAIN = 5000.0
KD_GAIN = 2 * np.sqrt(KP_GAIN)
robot._articulation_view.set_gains(KP_GAIN, KD_GAIN)

ego_camera_path = "/World/Franka/panda_hand/ego_camera_Xform/ego_camera"
exo_camera_path = '/Replicator/exo_camera_Xform/exo_camera'
with rep.new_layer():
    rep.create.camera(parent="/World/Franka/panda_hand", position=(0.2, -0.1, -0.9), name="ego_camera",
                      rotation=(-7.4, 80.5, -6.5))
    rep.create.camera(position=(-0.2, 1.3, 1.2), look_at=(-0.2, 0.6, 0.7), name="exo_camera", focal_length=14.0)
    rep.create.light(light_type="Dome", intensity=1000.0)

ego_camera_view = CameraView(prim_paths_expr=ego_camera_path, camera_resolution=(256, 256))
exo_camera_view = CameraView(prim_paths_expr=exo_camera_path, camera_resolution=(256, 256))

articulation_controller = robot.get_articulation_controller()
all_joint_names = [f"panda_joint{i + 1}" for i in range(7)] + ["panda_finger_joint1", "panda_finger_joint2"]
full_state_getter = ArticulationSubset(robot, all_joint_names)

# === 连接 ===
print(f"正在连接 WSL 推理服务 {SERVER_HOST}:{SERVER_PORT} ...")
client = RobotInferenceClient(host=SERVER_HOST, port=SERVER_PORT)

# === 循环 ===
my_world.reset()
my_world.step()
for _ in range(10): my_world.step(render=True)

print(f"开始任务: {TASK_DESC}")

while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        # 1. 获取
        ego_raw = ego_camera_view.get_rgb()[0].cpu().numpy()[:, :, :3]
        exo_raw = exo_camera_view.get_rgb()[0].cpu().numpy()[:, :, :3]
        full_pos = full_state_getter.get_joint_positions()

        # 2. 预处理 (发送 224x224)
        # 官方示例用的就是这个形状！
        obs = {
            "video.ego_view": np.expand_dims(ego_raw, axis=0),
            "video.exo_view": np.expand_dims(exo_raw, axis=0),

            # 状态也加 Batch
            "state.franka_arm": np.expand_dims(full_pos[:7], axis=0),
            "state.franka_hand": np.expand_dims(full_pos[7:], axis=0),
            "annotation.human.action.task_description": [TASK_DESC]  # 列表本身就是 batch
        }

        # 3. 推理
        try:
            action_dict = client.get_action(obs)
            # --- 提取手臂动作 (7维) ---
            # 1. 取出三部分数据 (只取第一帧 [0])
            pred_arm_6d = action_dict["action.franka_arm"][0]  # Shape (6,)
            pred_arm_rot = action_dict["action.franka_arm_rot"][0]  # Shape (1,)
            pred_hand = action_dict["action.franka_hand"][0]  # Shape (2,)

            # 2. 拼接：6D手臂 + 1D旋转 = 7D完整手臂
            target_arm_7d = np.concatenate([pred_arm_6d, pred_arm_rot])

            # 3. 再次拼接：7D手臂 + 2D夹爪 = 9D全关节
            full_target = np.concatenate([target_arm_7d, pred_hand])
            full_target = full_target * 10
            # 设置打印精度，方便查看
            np.set_printoptions(precision=3, suppress=True)
            seq_arm = action_dict["action.franka_arm"]

            # print("\n--- 轨迹检查 ---")
            # print(f"当前第 0 步目标: {seq_arm[0]}")
            # print(f"未来第15 步目标: {seq_arm[-1]}")
            #
            # diff = np.linalg.norm(seq_arm[-1] - seq_arm[0])
            # print(f"首尾差异(距离): {diff:.4f}")




            # 4. 执行
            action_apply = ArticulationAction(joint_positions=full_target)
            articulation_controller.apply_action(action_apply)
        except Exception as e:
            print(f"Error: {e}")

simulation_app.close()