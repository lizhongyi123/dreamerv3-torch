#obs, reward, done, info = env.step(action)
#输入是一个6维动作

#输出是
# obs,:{
#   'image': np.array(shape=(64, 64, 3), dtype=uint8),
#   'position': ...,
#   'velocity': ...,
#   'is_first': bool,
#   'is_terminal': bool,
# }
# reward：浮点数
#
# done（结束标志）
#
# 布尔值，仅在 episode 结束时 True


import envs.dmc as dmc
from envs.wrappers import NormalizeActions
import numpy as np
import matplotlib.pyplot as plt

# 1. 创建环境
env = dmc.DeepMindControl(
    name="walker_walk",
    action_repeat=2,
    size=(64, 64),
    seed=0,
)
env = NormalizeActions(env)  # 把动作空间归一化到 [-1, 1]

# 2. reset，拿到初始观测
obs = env.reset()

fall_action = np.array([1, 1, 1, 1, 1, 1], dtype=np.float32)

for t in range(100):
    obs, reward, done, info = env.step(fall_action)
    plt.imshow(obs["image"])
    plt.axis("off")
    plt.pause(1)
    if done:
        print("Walker 摔倒了！按回车继续看画面...")
        input()
        break



# print("初始 obs keys:", obs.keys())
# print("初始图像 shape:", obs["image"].shape)
#
# # 3. 查看动作空间信息
# action_space = env.action_space
# print("动作空间 shape:", action_space.shape)
# print("动作空间 low:", action_space.low)
# print("动作空间 high:", action_space.high)
#
# # 4. 构造一个“传递给环境的动作”
#
# ## (1) 全 0 动作（不动）
# zero_action = np.zeros(action_space.shape, dtype=np.float32)
# print("\nZero action:", zero_action)
#
# ## (2) 随机动作（在 [-1, 1] 内）
# random_action = np.random.uniform(-1.0, 1.0, size=action_space.shape).astype(np.float32)
# print("Random action:", random_action)
# random_action = np.array([0.99] * 6, dtype=np.float32)
# # 5. 把动作传入环境（先用随机动作做一步）
# obs2, reward, done, info = env.step(random_action)
#
# print("\n执行一步后的：")
# print("reward:", reward)
# print("done:", done)
# print("info:", info)
# print("下一步 obs keys:", obs2.keys())
# print("下一步图像 shape:", obs2["image"].shape)
#
# # 6. 把这一步之后的图像画出来
# plt.figure(figsize=(8, 4))  # 宽8高4的画布
#
# # 子图1：动作执行前的图像
# plt.subplot(1, 2, 1)
# plt.imshow(obs["image"])
# plt.title("Before action")
# plt.axis("off")
#
# # 子图2：动作执行后的图像
# plt.subplot(1, 2, 2)
# plt.imshow(obs2["image"])
# plt.title(f"After action, reward={reward:.3f}")
# plt.axis("off")
#
# plt.tight_layout()
# plt.show()
