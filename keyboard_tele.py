from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.sensors.camera import Camera
from isaacsim.core.api import World
import isaacsim.core.utils.numpy.rotations as rot_utils
import numpy as np
import matplotlib.pyplot as plt
# from isaacsim.robot_motion.motion_generation.articulation_kinematics_solver import ArticulationKinematicsSolver
# from isaacsim.robot_motion.motion_generation.kinematics_interface import KinematicsSolver
from isaacsim.robot.manipulators.examples.franka.kinematics_solver import KinematicsSolver
import carb
import numpy as np
from isaacsim.core.utils.types import ArticulationAction
from pxr import Gf, UsdGeom
import cv2
import time
#isaacsim.robot_motion.motion_generation

my_world = World(stage_units_in_meters=1.0)

import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.prims import SingleArticulation

# usd_path = "/home/user/table.usd"
# prim_path = "/World/envs/panda"

usd_path = "/home/user/FCloud_OmniBot赛事指定文件/赛道三：FrankaEmika/panda_instanceable.usd"
prim_path = "/World/envs/env_0/panda"

living_room_path = "/home/user/FCloud_OmniBot赛事指定文件/赛道三：table_clean/living_room_table/living_room_table.usd"
living_room_prim_path = "/World/envs/env_0/living_room"

tomato_sauce_path = "/home/user/FCloud_OmniBot赛事指定文件/赛道三：table_clean/tomato_sauce/tomato_sauce.usd"
tomato_sauce_prim_path = "/World/envs/env_0/tomato_sauce"

living_room = stage_utils.add_reference_to_stage(living_room_path, living_room_prim_path)
tomato = stage_utils.add_reference_to_stage(tomato_sauce_path, tomato_sauce_prim_path)
xformable = UsdGeom.Xformable(tomato)

for op in xformable.GetOrderedXformOps():
    print("op name:",op.GetOpName())
    if op.GetOpName() == "xformOp:scale":
        op.Set(Gf.Vec3f(0.02, 0.02, 0.02))  # 修改已有 scale
        break
    if op.GetOpName() == "xformOp:translate":
        op.Set(Gf.Vec3f(0.0, 0.0, 0.0))  # x=1, y=0, z=0.5
        break
    if op.GetOpName() == "xformOp:orient":
        op.Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))  # w,x,y,z
        break
else:
    # 如果没有 scale，则添加
    xformable.AddScaleOp().Set(Gf.Vec3f(0.02, 0.02, 0.02))

# load the Franka Panda robot USD file
robot = stage_utils.add_reference_to_stage(usd_path, prim_path)
# wrap the prim as an articulation
articulation_view = SingleArticulation(prim_path=prim_path, name="franka_panda")
articulation_view.initialize()

from isaacsim.core.prims import Articulation
from isaacsim.core.api.controllers.articulation_controller import ArticulationController
#articulation_controller = ArticulationController()
from isaacsim.core.prims import Articulation

# articulation_view = Articulation(
#     prim_paths_expr = prim_path,
#     name="panda_view",
#     reset_xform_properties=False,
# )
# print("DOF:",articulation_view._num_dof)

#articulation_controller.initialize(articulation_view)

camera = Camera(
    prim_path="/World/camera",
    position=np.array([1.0, 0.0, 0.5]),
    frequency=20,
    resolution=(512, 512),
    orientation=rot_utils.euler_angles_to_quats(np.array([-25, 65, 85]), degrees=True),
)

my_world.scene.add_default_ground_plane()
my_world.reset()
my_world.step(render=True)
camera.initialize()

i = 0
camera.add_motion_vectors_to_frame()

from isaacsim.robot.manipulators.examples.franka.kinematics_solver import KinematicsSolver
articulation_solver = KinematicsSolver(robot_articulation=articulation_view,end_effector_frame_name="panda_joint7")

STEP = 0.02
# current_pose = articulation_solver.compute_end_effector_pose()
# print("cur_pose:",current_pose)

#current_pose = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
def on_keyboard(event, *args):
    global current_pose
    print("cur:",current_pose)
    if event.event == carb.input.MouseEvent:
        return

    key = event.event.input
    print("key:",key)
    # 方向键控制
    if key == carb.input.KeyboardInput.Q:
        current_pose[0] += STEP
        action = ArticulationAction(joint_positions=np.array([current_pose[0]]), joint_indices=np.array([0]))
        articulation_controller.apply_action(action)
    elif key == carb.input.KeyboardInput.W:
        current_pose[0] -= STEP
        action = ArticulationAction(joint_positions=np.array([current_pose[0]]), joint_indices=np.array([0]))
        articulation_controller.apply_action(action)
    elif key == carb.input.KeyboardInput.E:
        current_pose[1] += STEP
        action = ArticulationAction(joint_positions=np.array([current_pose[1]]) , joint_indices=np.array([1]))
        articulation_controller.apply_action(action)
    elif key == carb.input.KeyboardInput.R:
        current_pose[1] -= STEP
        action = ArticulationAction(joint_positions=np.array([current_pose[1]]) , joint_indices=np.array([1]))
        articulation_controller.apply_action(action)
    elif key == carb.input.KeyboardInput.T:
        current_pose[2] += STEP
        action = ArticulationAction(joint_positions=np.array([current_pose[2]]) , joint_indices=np.array([2]))
        articulation_controller.apply_action(action)
    elif key == carb.input.KeyboardInput.Y:
        current_pose[2] -= STEP
        action = ArticulationAction(joint_positions=np.array([current_pose[2]]) , joint_indices=np.array([2]))
        articulation_controller.apply_action(action)
    elif key == carb.input.KeyboardInput.U:
        current_pose[3] += STEP
        action = ArticulationAction(joint_positions=np.array([current_pose[3]]) , joint_indices=np.array([3]))
        articulation_controller.apply_action(action)
    elif key == carb.input.KeyboardInput.I:
        current_pose[3] -= STEP
        action = ArticulationAction(joint_positions=np.array([current_pose[3]]) , joint_indices=np.array([3]))
        articulation_controller.apply_action(action)
    elif key == carb.input.KeyboardInput.O:
        current_pose[4] += STEP
        action = ArticulationAction(joint_positions=np.array([current_pose[4]]) , joint_indices=np.array([4]))
        articulation_controller.apply_action(action)
    elif key == carb.input.KeyboardInput.P:
        current_pose[4] -= STEP
        action = ArticulationAction(joint_positions=np.array([current_pose[4]]) , joint_indices=np.array([4]))
        articulation_controller.apply_action(action)
    elif key == carb.input.KeyboardInput.A:
        current_pose[5] += STEP
        action = ArticulationAction(joint_positions=np.array([current_pose[5]]) , joint_indices=np.array([5]))
        articulation_controller.apply_action(action)
    elif key == carb.input.KeyboardInput.S:
        current_pose[5] -= STEP
        action = ArticulationAction(joint_positions=np.array([current_pose[5]]) , joint_indices=np.array([5]))
        articulation_controller.apply_action(action)
    elif key == carb.input.KeyboardInput.D:
        current_pose[6] += STEP
        action = ArticulationAction(joint_positions=np.array([current_pose[6]]) , joint_indices=np.array([6]))
        articulation_controller.apply_action(action)
    elif key == carb.input.KeyboardInput.F:
        current_pose[6] -= STEP
        action = ArticulationAction(joint_positions=np.array([current_pose[6]]) , joint_indices=np.array([6]))
        articulation_controller.apply_action(action)
    # 抓手控制（假设两个关节）
    elif key == carb.input.KeyboardInput.Z:  # 张开
        action = ArticulationAction(joint_positions=np.array([0.3,0.3]) , joint_indices=np.array([7,8]))
        articulation_controller.apply_action(action)
    elif key == carb.input.KeyboardInput.X:  # 闭合
        action = ArticulationAction(joint_positions=np.array([0.3,0.3]) , joint_indices=np.array([7,8]))
        articulation_controller.apply_action(action)
    elif key == carb.input.KeyboardInput.M:  #take photo
        img = camera.get_rgba()[:, :, :3]
        image_name = "/home/user/IsaacLab/images/" + str(time.time()) + ".jpg"
        cv2.imwrite(image_name,img)
    elif key == carb.input.KeyboardInput.ESCAPE:
        simulation_app.close()

# input_interface = carb.input.acquire_input_interface()
# input_interface.subscribe_to_input_events(on_keyboard, order=0)

while simulation_app.is_running():
    my_world.step(render=True)
    target_pose = (np.array([0.0,0.0,0.0]),np.array([0.0,0.0,0.0,1.0]))
    success, joint_positions = articulation_solver.compute_inverse_kinematics(
        target_position=target_pose[0],
        target_orientation = target_pose[1]
    )

    if success:
        robot.set_joint_positions(joint_positions)

    
    #print(camera.get_current_frame())
    # if i == 100:
    #     imgplot = plt.imshow(camera.get_rgba()[:, :, :3])
    #     plt.show()
    #     print(camera.get_current_frame()["motion_vectors"])
    if my_world.is_playing():
        if my_world.current_time_step_index == 0:
            my_world.reset()
    i += 1


simulation_app.close()






