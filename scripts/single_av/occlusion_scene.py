import gymnasium as gym
import mani_skill.envs
import numpy as np
import sapien
import matplotlib.pyplot as plt
import torch


def create_env():
    """创建 ManiSkill 环境"""
    env = gym.make(
        "OpenCabinetDrawer-v1",
        obs_mode="rgbd",
        render_mode="human",
        control_mode="pd_ee_delta_pose",  # 使用末端执行器位姿控制
        # robot="panda",  # 使用 Panda 机械臂
        # reward_mode="dense",  # 使用密集奖励
        # max_episode_steps=200,  # 设置最大步数
        # # 设置桌面场景
        # cabinet_model="cabinet_1_drawer",  # 使用单抽屉柜子
        # cabinet_scale=1.0,  # 设置柜子大小
        # table_height=0.6,  # 设置桌面高度
    )
    obs, _ = env.reset()
    return env, obs


def find_drawer_link(cabinet, target_name="scene-0-1004-0_link_0"):
    """在 cabinet 中查找指定名的 link"""
    if not hasattr(cabinet, "_objs"):
        raise TypeError("cabinet 不包含 _objs 属性，可能不是有效的 Articulation 管理器")

    for art in cabinet._objs:
        try:
            link = art.find_link_by_name(target_name)
            if link:
                return link
        except Exception:
            continue
    return None


def create_occluder(scene, drawer_pose, size=(0.04, 0.08, 0.06), color=(0.2, 0.8, 0.2, 1.0)):
    """创建一个遮挡物盒子并放置到抽屉前方"""
    half_size = np.array(size)
    # 调整遮挡物位置，使其位于桌面上
    offset = np.array([0.15, 0, 0.03])  # 降低高度，使其位于桌面上
    position = drawer_pose.p + offset

    # 创建遮挡物
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=half_size)
    builder.add_box_visual(half_size=half_size, material=sapien.render.RenderMaterial(
        base_color=color
    ))
    occluder = builder.build(name="occluder_box")
    
    # 设置遮挡物位置
    occluder.set_pose(sapien.Pose(p=position))
    print(f"已在位置 {np.round(position, 3)} 放置遮挡物。")


def run_visualization_loop(env, obs):
    """主循环渲染机器人腕部视角图像"""
    plt.ion()
    fig, ax = plt.subplots()
    wrist_cam_rgb = obs["sensor_data"]["fetch_head"]["rgb"].cpu().numpy().squeeze(0)
    img_display = ax.imshow(wrist_cam_rgb)
    ax.set_title("Panda Wrist Camera View")
    plt.axis('off')

    # 初始化动作空间
    action = np.zeros(7)  # 6个自由度 + 1个抓取动作
    
    try:
        terminated = False
        truncated = False
        step = 0
        
        while not (terminated or truncated):
            # 简单的控制逻辑
            if step < 50:  # 前50步：移动到抽屉附近
                action = np.array([0.1, 0, 0, 0, 0, 0, 0])
            elif step < 100:  # 接下来50步：尝试抓取抽屉
                action = np.array([0, 0, 0, 0, 0, 0, 1])
            else:  # 最后尝试拉开抽屉
                action = np.array([0.2, 0, 0, 0, 0, 0, 1])
            
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            wrist_cam_rgb = obs["sensor_data"]["fetch_head"]["rgb"].cpu().numpy().squeeze(0)
            img_display.set_data(wrist_cam_rgb)
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            plt.pause(0.01)
            step += 1
            
    except Exception as e:
        print(f"可视化循环发生错误: {e}")
    finally:
        plt.ioff()
        plt.close(fig)


def setup_occlusion_scene():
    print("ManiSkill 3: 正在搭建桌面遮挡场景...")
    env, obs = create_env()

    # 替代逻辑获取场景和柜子对象
    try:
        scene = env.unwrapped.scene
        cabinet = env.unwrapped.cabinet
    except AttributeError:
        print("无法直接访问 scene 和 cabinet 属性，尝试通过其他方法获取...")
        # 添加其他获取逻辑（如果有）

    # 如果仍然无法获取，抛出错误
    if scene is None or cabinet is None:
        raise ValueError("无法获取场景或柜子对象，请检查环境配置或 API 文档。")

    print("正在查找抽屉拉手链接...")
    drawer_link = find_drawer_link(cabinet)
    if not drawer_link:
        raise ValueError("未找到指定名称的抽屉拉手链接，请检查名称或模型结构。")
    # 替换弃用的 get_pose 方法
    drawer_pose = drawer_link.get_entity_pose()
    print(f"抽屉拉手链接位置: {np.round(drawer_pose.p, 3)}")

    print("正在创建遮挡物...")
    create_occluder(scene, drawer_pose)

    print("开始主循环...")
    run_visualization_loop(env, obs)

    env.close()
    print("环境已关闭。")


if __name__ == "__main__":
    setup_occlusion_scene()
