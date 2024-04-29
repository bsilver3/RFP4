import gymnasium as gym
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import Adam
from collections import deque
from gymnasium.wrappers import FrameStack
import random
from tqdm import tqdm

# Define constants and hyperparameters
num_episodes = 10
max_steps_per_episode = 1000
learning_rate = 0.1 # Use 0.0001 for 10000 episodes, 0.001 for 1000 episodes, and 0.01 for 100 episodes
batch_size = 64
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.95 # Use 0.995 for 1000 episodes, 0.98 for 100 episodes, 
memory = deque(maxlen=25000)  # Experience replay buffer
env_name = "ALE/Frogger-v5"

def build_model(input_shape, num_actions):
    model = Sequential([ # Each person should change the amount of Conv2D/Dense layers, as well as the filter amount and kernel_size/strides
        Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape, data_format="channels_first"),
        Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
        Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_actions, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

# Create the environment
env = gym.make(env_name, obs_type='grayscale')
num_actions = env.action_space.n

env = FrameStack(env, 4)
frames, width, height = env.observation_space.shape

# Build the DQN model
model = build_model((frames, width, height), num_actions)

# Training loop
for episode in tqdm(range(num_episodes), desc='Episode Progress', position=0):
    state, _ = env.reset()
    episode_reward = 0
    done = False

    for step in range(max_steps_per_episode):
        if np.random.rand() <= epsilon:
            action = env.action_space.sample()  # Exploration
        else:
            q_values = model.predict(np.array([state]), verbose=None)[0]
            action = np.argmax(q_values)  # Exploitation

        # Ensure action is within bounds
        action = np.clip(action, 0, num_actions - 1)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward

        memory.append((state, action, reward, next_state, done))

        state = next_state

        if done:
            break

    # Experience replay
    if len(memory) >= batch_size:
        minibatch = random.sample(memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + gamma * np.amax(model.predict(np.array([next_state]), verbose=None)[0])

            target_f = model.predict(np.array([state]), verbose=None)
            target_f[0][action] = target
            model.fit(np.array([state]), target_f, epochs=1, verbose=None)

    # Decay exploration rate
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    print(f"\rPrevious episode: Episode: {episode + 1}/{num_episodes}, Total Reward: {episode_reward}, Epsilon: {epsilon:.4f}", end="")

env.close()

model.save_weights('mihail.weights.h5')