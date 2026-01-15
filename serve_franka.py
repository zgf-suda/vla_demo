import sys
import os
# ============ 1. 强制离线模式 (解决 ConnectionRefused) ============
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.append(r"/mnt/c/Users/xjy47/Desktop/SIM/Isaac-GR00T-main")

import torch
import numpy as np
from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.model.transforms import GR00TTransform
from gr00t.model.policy import Gr00tPolicy
from gr00t.eval.robot import RobotInferenceServer
from gr00t.data.schema import EmbodimentTag

MODEL_PATH = "/mnt/c/Users/xjy47/Desktop/SIM/checkpoint-1000/checkpoint-1000"

# ============ 配置 ============
video_keys = ["video.ego_view", "video.exo_view"]
video_modality = ModalityConfig(delta_indices=[0], modality_keys=video_keys)
state_modality = ModalityConfig(delta_indices=[0], modality_keys=["state.franka_arm", "state.franka_hand"])
action_modality = ModalityConfig(delta_indices=list(range(16)), modality_keys=["action.franka_arm", "action.franka_arm_rot", "action.franka_hand"])
language_modality = ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.action.task_description"])

modality_configs = {"video": video_modality, "state": state_modality, "action": action_modality, "language": language_modality}

# ============ 转换流程 (极简版) ============
# 既然 Client 发来的是标准 uint8 Numpy，这里什么都不用做，直接交给 GR00TTransform
transforms = ComposedModalityTransform(transforms=[
    # --- 视频处理：无！(移除所有手动处理) ---
    # GR00TTransform 会自动调用内部的 image processor 处理 (1, 256, 256, 3) 的数据

    # --- 状态/动作处理 (保持不变，归一化还是需要的) ---
    StateActionToTensor(apply_to=state_modality.modality_keys),
    StateActionTransform(apply_to=state_modality.modality_keys, normalization_modes={"state.franka_arm": "min_max", "state.franka_hand": "min_max"}),
    StateActionToTensor(apply_to=action_modality.modality_keys),
    StateActionTransform(apply_to=action_modality.modality_keys, normalization_modes={"action.franka_arm": "min_max", "action.franka_arm_rot": "min_max", "action.franka_hand": "min_max"}),

    # --- 拼接 ---
    ConcatTransform(video_concat_order=video_modality.modality_keys, state_concat_order=state_modality.modality_keys, action_concat_order=action_modality.modality_keys),
    
    # --- 模型输入 ---
    GR00TTransform(state_horizon=1, action_horizon=16, max_state_dim=64, max_action_dim=32),
])

if __name__ == "__main__":
    print(f"--- 正在 WSL 中加载模型 ---")
    policy = Gr00tPolicy(
        model_path=MODEL_PATH,
        modality_config=modality_configs,
        modality_transform=transforms,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        denoising_steps=4
    )
    print(">>> Server Ready: 5555")
    server = RobotInferenceServer(policy, port=5555)
    server.run()