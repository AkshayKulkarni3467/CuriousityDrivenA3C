import collections
import cv2
import numpy as np
import gymnasium as gym
import miniworld


class RepeatAction(gym.Wrapper):
    def __init__(self, env=None, repeat=4, fire_first=False):
        super(RepeatAction, self).__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.low.shape
        self.fire_first = fire_first

    def step(self, action):
        t_reward = 0.0
        done = False
        for i in range(self.repeat):
            obs, reward, terminated,truncated, info = self.env.step(action)
            done = terminated or truncated
            t_reward += reward
            if done:
                break
        return obs, t_reward, terminated,truncated, info

    def reset(self,seed= None,options= None):
        obs,lives = self.env.reset()
        if self.fire_first:
            assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
            obs, _, _,_, _ = self.env_step(1)
        return obs,lives


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, shape, env=None):
        super(PreprocessFrame, self).__init__(env)
        self.shape = (shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                                shape=self.shape,
                                                dtype=np.float32)

    def observation(self, obs):
        new_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized_screen = cv2.resize(new_frame, self.shape[1:],
                                    interpolation=cv2.INTER_AREA)
        new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
        new_obs = new_obs / 255.0

        return new_obs


class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, repeat):
        super(StackFrames, self).__init__(env)
        self.observation_space = gym.spaces.Box(
                env.observation_space.low.repeat(repeat, axis=0),
                env.observation_space.high.repeat(repeat, axis=0),
                dtype=np.float32)
        self.stack = collections.deque(maxlen=repeat)

    def reset(self,seed = None,options = None):
        self.stack.clear()
        observation,lives = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape),lives

    def observation(self, observation):
        self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)


def make_env(env_name, shape=(42, 42, 1), repeat=4,render_mode = 'human'):
    env = gym.make(env_name,render_mode = render_mode)
    env = RepeatAction(env, repeat)
    env = PreprocessFrame(shape, env)
    env = StackFrames(env, repeat)
    return env
