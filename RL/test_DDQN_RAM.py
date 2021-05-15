from RL.wrappers import *
import torch
# import time

SAVED_MODELS_FILE = "./saved_models/"
MODEL_TO_TEST = "Boxing-ram-v42000000_short.pt"
ENV_NAME = 'Boxing-ram-v4'

env = gym.make(ENV_NAME)
env = PytorchRAMWrapper(env, add_done=True)

N_ACTIONS = env.action_space.n  # taken from env

image = env.reset()
image = torch.tensor(image)

model = torch.load(SAVED_MODELS_FILE + MODEL_TO_TEST, map_location=torch.device('cpu'))
model.eval()

total_reward = 0
total_score = 0
total_total_reward = 0
total_total_score = 0
done = False
for i in range(100):
    total_reward = 0
    total_score = 0

    while not done:
        actions = model(image.unsqueeze(0).float() / 255.)
        action = torch.argmax(actions)
        action = action.detach().cpu().numpy()

        image, reward, done, info, _ = env.step(action)
        image = torch.tensor(image)
        # env.render()
        # time.sleep(0.1)

        total_reward += reward
        if done:
            image = env.reset()
            image = torch.tensor(image)
            total_score = info["score"]

    total_total_reward = total_total_reward + total_reward
    total_total_score = total_total_score + total_score
    done = False


print("avg reward: " + str(total_total_reward / 100) + ", avg score: " + str(total_total_score / 100))
