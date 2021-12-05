import os.path
import time

import gym
import numpy as np
import torch
from tqdm import tqdm

from models.PolicyGradientAgent import PolicyGradientAgent
from models.PolicyGradientNetwork import PolicyGradientNetwork
from util import fix

start = time.time()

game = gym.make('LunarLander-v2')

fix(game)
print("*** observation 输出")
print(game.observation_space)
print("*** action_space 输出")
print(game.action_space)
initial_state = game.reset()
print("*** 环境初始化完成")
print(initial_state)

# 小实验
# game.reset()
#
# img = plt.imshow(game.render(mode='rgb_array'))
#
# done = False
# while not done:
#     action = game.action_space.sample()
#     observation, reward, done, _ = game.step(action)
#
#     img.set_data(game.render(mode='rgb_array'))
#     display.display(plt.gcf())
#     display.clear_output(wait=True)

network = PolicyGradientNetwork()
agent = PolicyGradientAgent(network)
#
agent.network.train()  # 訓練前，先確保 network 處在 training 模式
EPISODE_PER_BATCH = 5  # 每蒐集 5 個 episodes 更新一次 agent
NUM_BATCH = 400  # 總共更新 400 次
EXP_NAME = "exp_1"

#
avg_total_rewards, avg_final_rewards = [], []

prg_bar = range(NUM_BATCH)
for batch in tqdm(prg_bar):

    log_probs, rewards = [], []
    total_rewards, final_rewards = [], []

    # 蒐集訓練資料
    for episode in range(EPISODE_PER_BATCH):

        state = game.reset()
        total_reward, total_step = 0, 0
        seq_rewards = []
        while True:

            action, log_prob = agent.sample(state)  # at , log(at|st)
            next_state, reward, done, _ = game.step(action)

            log_probs.append(log_prob)  # [log(a1|s1), log(a2|s2), ...., log(at|st)]
            # seq_rewards.append(reward)
            state = next_state
            total_reward += reward
            total_step += 1
            rewards.append(reward)  # 改這裡
            # ! 重要 ！
            # 現在的reward 的implementation 為每個時刻的瞬時reward, 給定action_list : a1, a2, a3 ......
            #                                                       reward :     r1, r2 ,r3 ......
            # medium：將reward調整成accumulative decaying reward, 給定action_list : a1,                         a2,                           a3 ......
            #                                                       reward :     r1+0.99*r2+0.99^2*r3+......, r2+0.99*r3+0.99^2*r4+...... ,r3+0.99*r4+0.99^2*r5+ ......
            # boss : implement DQN
            if done:
                final_rewards.append(reward)
                total_rewards.append(total_reward)
                break

    # print(f"rewards looks like ", np.shape(rewards))
    # print(f"log_probs looks like ", np.shape(log_probs))
    # 紀錄訓練過程
    avg_total_reward = sum(total_rewards) / len(total_rewards)
    avg_final_reward = sum(final_rewards) / len(final_rewards)
    avg_total_rewards.append(avg_total_reward)
    avg_final_rewards.append(avg_final_reward)
    print("avg_total_reward: " + str(avg_total_reward) + " avg_final_reward: " + str(avg_final_reward))
    # prg_bar.set_description(f"Total: {avg_total_reward: 4.1f}, Final: {avg_final_reward: 4.1f}")

    # 更新網路
    # rewards = np.concatenate(rewards, axis=0)
    rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)  # 將 reward 正規標準化
    agent.learn(torch.stack(log_probs), torch.from_numpy(rewards))
    # print("logs prob looks like ", torch.stack(log_probs).size())
    # print("torch.from_numpy(rewards) looks like ", torch.from_numpy(rewards).size())
save_path = os.path.join('./experiments', EXP_NAME)
agent.save(save_path)
