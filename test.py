import gymnasium as gym
import miniworld
from actor_critic import ActorCritic
from icm import ICM
import torch as T
import numpy as np
from wrappers import make_env
import time
from utils import plot_learning_curve


env = make_env('MiniWorld-Hallway-v0',render_mode = 'human')

episode = 0
episodes = 100
n_actions = 3
input_shape = [4, 42, 42]
icm = True
scores = []


agent = ActorCritic(input_shape,n_actions)
agent.load_state_dict(T.load('checkpoints/saved_model.pth'))
while episode < episodes:
    done = False
    state,_ = env.reset()
    score = 0
    hx = T.zeros(1, 256)
    while not done:
        env.render()
        state = T.tensor(state[np.newaxis,:], dtype=T.float)
        action,_,_,_ = agent(state,hx)
        # action = env.action_space.sample()
        next_state,reward,terminated,truncated,info = env.step(action)
        score+=reward
        done = terminated or truncated
        state = next_state
        time.sleep(0.05)
    scores.append(score)
    print(f'Episode : {episode} Average reward (100) : {np.round(np.mean(scores[-100:]),2)}')
    episode+=1

x = [z for z in range(episode)]
plot_learning_curve(x, scores, 'ICM_hallway_sample.png')

env.close()