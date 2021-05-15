import gym
import numpy as np
import cv2


class PytorchWrapper(gym.Wrapper):
    # adopted from atari wrappers
    # https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
    # changes returns from atari gym object
    # resize and greyscale, stack k frames and clip rewards

    def __init__(self, env, k, size=(84, 84), add_done=False):
        super(PytorchWrapper, self).__init__(env)
        self.add_done = add_done
        self.frame_stack = 0
        self.k = k
        self.last_life_count = 0
        self.score = 0
        self.size = size

    def reset(self):
        self.score = 0
        self.last_life_count = 0

        ob = self.env.reset()
        ob = self.preprocess_frame(ob)

        # stack k times the reset ob
        self.frame_stack = np.stack([ob for i in range(self.k)])

        return self.frame_stack

    def step(self, action):
        reward = 0
        done = False
        additional_done = False

        # do k frame skips
        frames = []
        for i in range(self.k):

            ob, r, d, info = self.env.step(action)

            # if agents loses a life, insert additional done
            if self.add_done:
                if info['ale.lives'] < self.last_life_count:
                    additional_done = True
                self.last_life_count = info['ale.lives']

            ob = self.preprocess_frame(ob)
            frames.append(ob)

            reward += r

            if d:
                done = True
                break

        # shift observation to pytorch compatible input
        self.prepare_frame_stack(frames)

        # add score to info
        self.score += reward
        if done:
            info["score"] = self.score

        # clip the reward (-1, 1)
        if reward > 0:
            reward = 1
        elif reward == 0:
            reward = 0
        else:
            reward = -1

        return self.frame_stack, reward, done, info, additional_done

    def prepare_frame_stack(self, frames):

        num_frames = len(frames)

        # stack frames
        if num_frames == self.k:
            self.frame_stack = np.stack(frames)
        elif num_frames > self.k:
            self.frame_stack = np.array(frames[-self.k::])
        else:

            # change dims to pytorch compatible format
            self.frame_stack[0: self.k - num_frames] = self.frame_stack[num_frames::]
            # add frames to stack
            self.frame_stack[self.k - num_frames::] = np.array(frames)

    def preprocess_frame(self, ob):
        # resize and greyscale

        ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
        ob = cv2.resize(ob, dsize=self.size)

        return ob


class PytorchRAMWrapper(gym.Wrapper):
    # adopted from atari wrappers
    # https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
    # changes returns from atari gym object
    # resize and greyscale, stack k frames and clip rewards

    def __init__(self, env, add_done=False):
        super(PytorchRAMWrapper, self).__init__(env)
        self.add_done = add_done
        self.last_life_count = 0
        self.score = 0

    def reset(self):
        self.score = 0
        self.last_life_count = 0
        ob = self.env.reset()
        return ob

    def step(self, action):
        reward = 0
        done = False
        additional_done = False

        ob, r, d, info = self.env.step(action)

        # if agents loses a life, insert additional done
        if self.add_done:
            if info['ale.lives'] < self.last_life_count:
                additional_done = True
            self.last_life_count = info['ale.lives']

        reward += r

        if d:
            done = True

        # add score to info
        self.score += reward
        if done:
            info["score"] = self.score

        # clip the reward (-1, 1)
        if reward > 0:
            reward = 1
        elif reward == 0:
            reward = 0
        else:
            reward = -1

        return ob, reward, done, info, additional_done
