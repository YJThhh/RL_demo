import os
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
from IPython import display

from models.PolicyGradientAgent import PolicyGradientAgent
from models.PolicyGradientNetwork import PolicyGradientNetwork

start = time.time()
EXP_NAME = "exp_1"

print("*** 开始测试")
game = gym.make('LunarLander-v2')
network = PolicyGradientNetwork()
agent = PolicyGradientAgent(network)
save_path = os.path.join('./experiments', EXP_NAME, "model.pth")
agent.load(save_path)

agent.network.eval()  # 測試前先將 network 切換為 evaluation 模式
NUM_OF_TEST = 500  # Do not revise it !!!!!
test_total_reward = []
action_list = []
for i in range(NUM_OF_TEST):
    actions = []
    state = game.reset()

    # img = plt.imshow(game.render(mode='rgb_array'))
    total_reward = 0

    done = False
    while not done:
        action, _ = agent.sample(state)
        actions.append(action)
        state, reward, done, _ = game.step(action)

        total_reward += reward

        # img.set_data(game.render(mode='rgb_array'))
        # display.display(plt.gcf())
        # display.clear_output(wait=True)
    print(total_reward)
    test_total_reward.append(total_reward)

    action_list.append(actions)  # 儲存你測試的結果
    print("length of actions is ", len(actions))

print(f"Your final reward is : %.2f"%np.mean(test_total_reward))