import argparse
from ast import parse
from typing import Annotated
import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R
import sapien.core as sapien
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs.types import SimConfig
import os
from mani_skill.utils.structs import Actor, Link
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.motionplanner_stick import \
    PandaStickMotionPlanningSolver
import sapien.utils.viewer
import h5py
import json
import mani_skill.trajectory.utils as trajectory_utils
from mani_skill.utils import sapien_utils
from mani_skill.utils.wrappers.record import RecordEpisode
import tyro
from dataclasses import dataclass
import imageio.v2 as imageio
# 1. 新增 Open3D 的导入
import open3d as o3d

@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "occ_open_drawer"
    obs_mode: str = "rgb+depth+pointcloud"
    robot_uid: Annotated[str, tyro.conf.arg(aliases=["-r"])] = "panda"
    record_dir: str = "toy_example_record"
    save_video: bool = False
    viewer_shader: str = "rt-fast"
    video_saving_shader: str = "rt-fast"

def parse_args() -> Args:
    return tyro.cli(Args)

# =================================================================================
#  最终修正版的 save_data_for_afforddp 函数
# =================================================================================
# =================================================================================
#  根据您的提示再次修正的 save_data_for_afforddp 函数
# =================================================================================
# =================================================================================
#  针对 IndexError 最终修正版的 save_data_for_afforddp 函数
# =================================================================================

def save_data_for_afforddp(obs, env, output_dir="ipc_data"):
    """
        获取 ManiSkill 的观测数据并将其保存到文件, 供 afforddp 脚本使用。
        (最终修正版：使用 .squeeze() 自动处理批处理维度)
    """
    print(f"正在为 affordance transfer 保存数据至: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    CAMERA_NAME = 'hand_camera'

    # --- 1. 获取所有原始数据 ---
    if CAMERA_NAME not in obs['sensor_data']:
        raise KeyError(f"在 obs['sensor_data'] 中找不到相机 '{CAMERA_NAME}'。")

    sensor_obs = obs['sensor_data'][CAMERA_NAME]
    cam_param = obs['sensor_param'][CAMERA_NAME]

    print(f"相机参数: {cam_param}")

    # .squeeze() 会自动处理 (1, H, W, C) -> (H, W, C) 的情况，比 [0] 更安全
    cam_intrinsic_mat = cam_param["intrinsic_cv"].cpu().numpy().squeeze()
    cam_extrinsic_mat = cam_param["extrinsic_cv"].cpu().numpy().squeeze()
    cam_extrinsic_cv = cam_param["extrinsic_cv"].cpu().numpy().squeeze()

    # 补 1 行 → 4×4
    T_w2c = np.eye(4)
    T_w2c[:3, :4] = cam_extrinsic_cv

    # ① 先求逆，得到 camera→world
    T_c2w = np.linalg.inv(T_w2c)

    rgb_image = sensor_obs['rgb'].cpu().numpy().squeeze()
    depth_image = sensor_obs['depth'].cpu().numpy().squeeze()
    depth_image = depth_image.view(np.uint16)
    depth_image = depth_image.astype(np.float32) / 1000.0  # 将单位从毫米换成米
    depth_flat = depth_image.reshape(-1)

    # —— 在这里加入检测代码 —— 
    print(f"深度图范围：min = {depth_image.min():.4f} m, max = {depth_image.max():.4f} m")
    n_neg = np.sum(depth_image < 0)
    n_zero = np.sum(depth_image == 0)
    n_nan = np.isnan(depth_image).sum()
    n_inf = np.isinf(depth_image).sum()
    print(f"负值数量：{n_neg}，零值数量：{n_zero}，NaN 数量：{n_nan}，无限大数量：{n_inf}")
    # —— 检测结束 —— 

    seg_id_img = sensor_obs['segmentation'].cpu().numpy().squeeze() # H * W
    
    # 确保即使只有一个环境，RGB图像的维度也是正确的 (H, W, C)
    if rgb_image.ndim == 2: # 如果squeeze把通道也去掉了 (灰度图)
        rgb_image = np.expand_dims(rgb_image, axis=-1)

    H, W = depth_image.shape
    
    # --- 2. 从深度图(Depth)生成点云(PointCloud) ---
    y, x = np.mgrid[0:H, 0:W]
    uv_grid = np.stack([x, y], axis=-1)
    uv_flat = uv_grid.reshape(-1, 2)
    depth_flat = depth_image.reshape(-1)

    fx, fy = cam_intrinsic_mat[0, 0], cam_intrinsic_mat[1, 1]
    cx, cy = cam_intrinsic_mat[0, 2], cam_intrinsic_mat[1, 2]

    pcd_x = (uv_flat[:, 0] - cx) / fx * depth_flat
    pcd_y = (uv_flat[:, 1] - cy) / fy * depth_flat
    pcd_z = depth_flat
    
    pcd_cam_frame = np.stack([pcd_x, pcd_y, pcd_z], axis=-1)
    pcd_cam_homogeneous = np.hstack((pcd_cam_frame, np.ones((len(pcd_cam_frame), 1))))
    # pcd_world_homogeneous = (cam_extrinsic_mat @ pcd_cam_homogeneous.T).T
    pcd_world_homogeneous = (T_c2w @ pcd_cam_homogeneous.T).T[:, :3]  # 只取前4列
    points_xyz = pcd_world_homogeneous[:, :3]
    points_local_xyz = pcd_cam_homogeneous[:, :3]  # 相机坐标系下的点云
    points_rgb = rgb_image.reshape(-1, 3)

    # --- 3. 获取目标的分割掩码 ---
    TARGET_LINK_NAME = "link_2"
    target_seg_id = -1
    
    for seg_id, obj in env.unwrapped.segmentation_id_map.items():
        # if isinstance(obj, Link):
        #     print(f"分割ID: {seg_id}, Link名称: {obj.get_name()}")
        if isinstance(obj, Link) and TARGET_LINK_NAME in obj.get_name():
            target_seg_id = seg_id
            print(f"找到目标link '{obj.get_name()}'，其分割ID (per_scene_id) 为: {target_seg_id}")
            break

    if target_seg_id == -1:
        raise ValueError(f"无法在 segmentation_id_map 中找到包含名称 '{TARGET_LINK_NAME}' 的link。")

    seg_id_pcd = seg_id_img.reshape(-1)
    mask_env = seg_id_pcd.reshape(-1, 1)
    mask = (seg_id_pcd == target_seg_id)
    formatted_mask = mask.reshape(-1, 1).astype(int) * 4

    # --- 4. 保存所有处理好的数据到文件 ---
    np.save(os.path.join(output_dir, "cam_proj.npy"), cam_intrinsic_mat)
    np.save(os.path.join(output_dir, "cam_view.npy"), cam_extrinsic_mat)
    imageio.imwrite(os.path.join(output_dir, "rgb_image.png"), rgb_image)
    # save seg_id_img as image
    imageio.imwrite(os.path.join(output_dir, "segmentation_image.png"), seg_id_img.astype(np.uint8))
    np.save(os.path.join(output_dir, "depth_image.npy"), depth_image)
    np.save(os.path.join(output_dir, "points_xyz.npy"), points_xyz)
    # np.save(os.path.join(output_dir, "points_xyz.npy"), points_local_xyz)
    np.save(os.path.join(output_dir, "points_rgb.npy"), points_rgb)
    # np.save(os.path.join(output_dir, "formatted_mask.npy"), formatted_mask)
    np.save(os.path.join(output_dir, "formatted_mask.npy"), mask_env)

    metadata = {"cam_w": W, "cam_h": H}
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f)

    with open(os.path.join(output_dir, "data_ready.flag"), 'w') as f:
        f.write('ready')

    print(f"所有数据成功保存。点云包含 {points_xyz.shape[0]} 个点。")

    # =======================================================
    # 2. [新增] 使用 Open3D 可视化或保存点云
    # =======================================================
    print("正在准备使用 Open3D 处理点云...")

    pcd_full = o3d.geometry.PointCloud()
    pcd_full.points = o3d.utility.Vector3dVector(points_xyz)
    pcd_full.colors = o3d.utility.Vector3dVector(points_rgb.astype(np.float64) / 255.0)
    print(f"已生成完整点云，包含 {len(pcd_full.points)} 个点。")

    pts = np.asarray(pcd_full.points)
    print("Full point cloud bounds:")
    print("min:", pts.min(axis=0))
    print("max:", pts.max(axis=0))


    # =======================================================
    # 3. [新增] 裁切点云
    # =======================================================
    print("正在定义裁切范围...")
    # 定义一个轴对齐包围盒（AABB）的边界
    min_bound = np.array([-2, -2, -0.1])  # X, Y, Z 的最小值
    max_bound = np.array([2, 2, 2])   # X, Y, Z 的最大值

    # 创建包围盒对象
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    
    # 使用包围盒裁切点云
    pcd_cropped = pcd_full.crop(bbox)
    print(f"裁切后剩余 {len(pcd_cropped.points)} 个点。")


    # --- 4. 获取目标的分割掩码 (代码逻辑不变, 但作用于裁切后的点云) ---
    # 注意：为了逻辑简单，我们仍然在完整点云上计算mask，然后筛选裁切后的点
    TARGET_LINK_NAME = "link_2"
    target_seg_id = -1
    for seg_id, obj in env.unwrapped.segmentation_id_map.items():
        if isinstance(obj, Link) and TARGET_LINK_NAME in obj.get_name():
            target_seg_id = seg_id
            break
    if target_seg_id == -1:
        raise ValueError(f"无法在 segmentation_id_map 中找到包含名称 '{TARGET_LINK_NAME}' 的link。")
    seg_id_pcd = seg_id_img.reshape(-1)
    mask = (seg_id_pcd == target_seg_id)
    # formatted_mask = mask.reshape(-1, 1).astype(int) * 4 # 这个mask现在对应于完整点云

    # --- 5. 可视化或保存点云 (现在我们将使用裁切后的点云 'pcd_cropped') ---
    
    # 
    ply_filepath = os.path.join(output_dir, "scene_point_cloud_cropped.ply")
    print(f"将裁切后的点云保存至: {ply_filepath}")
    o3d.io.write_point_cloud(ply_filepath, pcd_cropped)

    # 可视化裁切后的点云，并把包围盒也画出来，方便调试
    bbox.color = (1, 0, 0) # 将包围盒颜色设为红色
    o3d.visualization.draw_geometries(
        [pcd_cropped, bbox], 
        window_name="Cropped Point Cloud with BBox"
    )
    
    # 如果您还想看下原始的完整点云是什么样的，可以取消下面这行的注释
    o3d.visualization.draw_geometries([pcd_full], window_name="Full Point Cloud")

def main(args: Args):
    # 此处省略了 main 函数中未改动的部分，以节省空间。
    # 您只需将上面的 save_data_for_afforddp 函数 和本文件顶部的 import 声明
    # 替换掉您现有代码中的对应部分即可。
    output_dir = f"{args.record_dir}/{args.env_id}/{args.robot_uid}"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/rgb", exist_ok=True)
    os.makedirs(f"{output_dir}/depth", exist_ok=True)
    os.makedirs(f"{output_dir}/videos", exist_ok=True)
    os.makedirs(f"{output_dir}/pcd", exist_ok=True)
    print(f"Output directory: {output_dir}")

    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        num_envs=1,
        control_mode="pd_joint_pos",
        render_mode="human",
        reward_mode="none",
        enable_shadow=True,
        viewer_camera_configs=dict(shader_pack=args.viewer_shader)
    )

    env = RecordEpisode(
        env,
        output_dir=output_dir,
        trajectory_name="trajectory",
        save_video=False,
        info_on_video=False,
        source_type="teleoperation",
        source_desc="teleoperation via the click+drag system"
    )

    seed = 10
    obs, info = env.reset(seed=seed)

    viewer = env.render_human()

    planner = PandaArmMotionPlanningSolver(
            env,
            debug=False,
            vis=True,
            base_pose=env.unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
            joint_acc_limits=0.5,
            joint_vel_limits=0.5,
        )

    '''
        STEP 1: Move arm to look for target affordance
    '''
    print("========Step 1: Move arm to look for target affordance==============")
    tcp_pose = env.unwrapped.agent.tcp.pose
    print(f"Current TCP Pose: {tcp_pose}")

    # delta_pos = np.array([-0.26, 0.1, 0.8])
    delta_pos = np.array([-0.40, 0.1, 0.8])
    target_pos = tcp_pose.raw_pose[0][:3].cpu().numpy() + delta_pos
    original_quat = tcp_pose.raw_pose[0][3:7].cpu().numpy()
    scipy_quat = np.array([original_quat[1], original_quat[2], original_quat[3], original_quat[0]])
    r_delta = R.from_rotvec(np.deg2rad(90) * np.array([1, 0, 0])) * R.from_rotvec(np.deg2rad(-60) * np.array([0, 1, 0])) * R.from_rotvec(np.deg2rad(-10) * np.array([0, 0, 1]))
    r_orig = R.from_quat(scipy_quat)
    r_combined = r_delta * r_orig
    combined_quat_xyzw = r_combined.as_quat()
    combined_quat = np.array([combined_quat_xyzw[3], combined_quat_xyzw[0], combined_quat_xyzw[1], combined_quat_xyzw[2]])
    target_pose = sapien.Pose(p=target_pos, q=combined_quat)
    print(f"Target Pose: {target_pose}")

    result = planner.move_to_pose_with_screw(target_pose, dry_run=True)
    if result != -1 and len(result["position"]) < 150:
        _, reward, _ ,_, info = planner.follow_path(result)
        print(f"Reward: {reward}, Info: {info}")
        print("Move to target pose successfully")
    else:
        if result == -1: print("Plan failed")
        else: print("Generated motion plan was too long. Try a closer sub-goal")

    '''
        STEP 2: Get current scene image and depth data
        1. 获取当前相机的图像和深度数据
        2. 保存数据到指定目录
        3. 使用 Open3D 可视化点云
    '''
    print("========Step 2: Get current scene image and depth data==============")
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    IPC_DIR = "ipc_data"
    save_data_for_afforddp(obs, env, output_dir=IPC_DIR)


    '''
        STEP 3: Get Affordance in this scene.
        1. 使用 Affordance Transfer 脚本获取当前场景的可用性数据
        2. 读取 affordance 数据 并调整机械臂位姿
    '''
    print("========Step 3: Get Affordance in this scene==============")
    env.close()

if __name__ == "__main__":
    args = parse_args()

    print("Parsed arguments:", args)
    
    main(args)