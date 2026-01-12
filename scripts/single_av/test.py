import gymnasium as gym
import mani_skill.envs  # 确保注册
import numpy as np
import imageio.v2 as imageio
import os

# 创建保存目录
os.makedirs("rgb", exist_ok=True)
os.makedirs("depth", exist_ok=True)

# 初始化环境
env = gym.make(
    "occ_open_drawer", 
    render_mode="human", 
    num_envs=1,
    obs_mode="rgbd", 
    control_mode="pd_ee_delta_pose"
)
# 重置环境
obs, info = env.reset()

# # 打印环境信息
# print("Observation space:", env.observation_space)
# print("Action space:", env.action_space)
# # 打印初始观测
# print("Initial observation:", obs)


step_count = 0

for cnt in range(10000):
    if cnt % 50 == 0:
        # 采样动作
        action = env.action_space.sample()

        print(f"Step {step_count}: Action sampled: {action}")
    else:
        # 使用零动作
        action = np.zeros(env.action_space.shape, dtype=np.float32)
    
    # 执行动作
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # # ====== 获取腕部相机图像 ======
    rgb = obs['sensor_data']['hand_camera']['rgb'][0].cpu().numpy()  # (128,128,3)
    depth = obs['sensor_data']['hand_camera']['depth'][0].cpu().numpy().squeeze()  # (128,128)

    imageio.imwrite(f"rgb/rgb_{step_count:05d}.png", rgb)
    np.save(f"depth/depth_{step_count:05d}.npy", depth)

    if done:
        print("Episode done. Resetting.")
        obs, info = env.reset()

    env.render()
    step_count += 1
# 关闭环境
env.close()
# 打印完成信息
print(f"Saved {step_count} frames to 'rgb' and 'depth' directories.")
