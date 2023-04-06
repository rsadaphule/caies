import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import gym
import random
from IPython import display
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Blackjack Environment, Helper functions, and other necessary functions should be placed here
# ...
BUFFER_SIZE = int(1e3)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 3e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from IPython import display

env = gym.make("Blackjack-v1")
env.observation_space

class JupyterDisplay(object):
    def __init__(self, figsize: tuple):
        self.figsize = figsize
        self.mode = "rgb_array"
    
    def show(self, env):
        plt.figure(figsize=self.figsize)
        plt.imshow(env.render())               # Removed render mode for compatibility
        plt.axis('off')
        display.clear_output(wait=True)
        display.display(plt.gcf())


def get_state_idxs(state):
    idx1, idx2, idx3 = state
    idx3 = int(idx3)
    return idx1, idx2, idx3


def update_qtable(qtable, state, action, reward, next_state, alpha, gamma):
    curr_idx1, curr_idx2, curr_idx3 = get_state_idxs(state)
    next_idx1, next_idx2, next_idx3 = get_state_idxs(next_state)
    curr_state_q = qtable[curr_idx1][curr_idx2][curr_idx3]
    next_state_q = qtable[next_idx1][next_idx2][next_idx3]
    qtable[curr_idx1][curr_idx2][curr_idx3][action] += \
            alpha * (reward + gamma * np.max(next_state_q) - curr_state_q[action])
    return qtable


def get_action(qtable, state, epsilon):
    if random.uniform(0,1) < epsilon:
        action = env.action_space.sample()
    else:
        idx1, idx2, idx3 = get_state_idxs(state)
        action = np.argmax(qtable[idx1][idx2][idx3])
    return action


FIGSIZE = (8,4)

def watch_trained_agent(env, qtable, num_rounds):
    envdisplay = JupyterDisplay(figsize=FIGSIZE)
    rewards = []
    for s in range(1, num_rounds+1):
        state, _ = env.reset()
        done = False
        round_rewards = 0
        while True:
            action = get_action(qtable, state, epsilon)          
            new_state, reward, done, _, info = env.step(action)  # Added blank for extra returned argument
            envdisplay.show(env)

            round_rewards += reward
            state = new_state
            if done == True:
                break
        rewards.append(round_rewards)
    return rewards


FIGSIZE = (8,4)

def watch_trained_agent_no_exploration(env, qtable, num_rounds):
    envdisplay = JupyterDisplay(figsize=FIGSIZE)
    rewards = []
    for s in range(1, num_rounds+1):
        state, _ = env.reset()
        done = False
        round_rewards = 0
        while True:
            action = get_action(qtable, state, 0)                # epsilon set to 0
            new_state, reward, done, _, info = env.step(action)  # Added blank for extra returned argument
            envdisplay.show(env)

            round_rewards += reward
            state = new_state
            if done == True:
                break
        rewards.append(round_rewards)
    return rewards


def print_policy(qtable):
    print('PC DC Soft Pol')
    dim1, dim2, dim3, dim4 = qtable.shape
    for player_count in range(10,21):
        for dealer_card in range(dim2):
            for soft in range(dim3):
                q_stay = qtable[player_count, dealer_card, soft, 0]
                q_hit  = qtable[player_count, dealer_card, soft, 1]
                pol = "Stay" if q_stay>=q_hit else "Hit"
                print(player_count+1, dealer_card+1, soft, pol)


class QNetwork(nn.Module):
    """
    -------
    Neural Network Used for Agent to Approximate Q-Values
    -------
    [Params]
        'state_size' -> size of the state space
        'action_size' -> size of the action space
        'seed' -> used for random module
    """
    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    
    
class Agent():
    """
    --------
    Deep Q-Learning Agent
    --------
    [Params]
        'state_size' -> size of the state space
        'action_size' -> size of the action space
        'seed' -> used for random module
    --------
    """
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """
        --------------
        Take an action given the current state (S(i))
        --------------
        [Params]
            'state' -> current state
            'eps' -> current epsilon value
        --------------
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state).cpu().data.numpy()
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values), np.max(action_values)
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):

        states, actions, rewards, next_states, dones = experiences

        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + gamma * q_targets_next * (1 - dones)
        q_expected = self.qnetwork_local(states).gather(1, actions)
        
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model: nn.Module, target_model: nn.Module, tau: float):
        """
        --------
        Update our target network with the weights from the local network
        --------
        Formula for each param (w): w_target = τ*w_local + (1 - τ)*w_target
        See https://arxiv.org/pdf/1509.02971.pdf
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

            
from dataclasses import dataclass

@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool

class ReplayBuffer:
    """
    ------------
    Used to store agent experiences
    ------------
    [Params]
        'action_size' -> length of the action space
        'buffer_size' -> Max size of our memory buffer
        'batch_size' -> how many memories to randomly sample
        'seed' -> seed for random module
    ------------
    """
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)            
            

from collections import deque

def dqn(n_episodes=2000, eps_start=0.99, eps_end=0.02, eps_decay=0.995):
    """
    -------------
    Train a Deep Q-Learning Agent
    -------------
    [Params]
        'n_episodes' -> number of episodes to train for
        'eps_start' -> epsilon starting value
        'eps_end' -> epsilon minimum value
        'eps_decay' -> how much to decrease epsilon every iteration
    -------------
    """

    scores = []                        
    scores_window = deque(maxlen=100)  
    eps = eps_start                   
    
    for episode in range(1, n_episodes+1):
        done = False
        episode_score = 0
        
        state, _ = env.reset()                                 # Added _ for new version of gym
        state = np.array(get_state_idxs(state), dtype=float)
        state[0] = state[0]/32
        state[1] = state[1]/10
        
        while not done:
            action = agent.act(state, eps)
            if isinstance(action, tuple):
                action, value = action
            else:
                value = 1.
            next_state, reward, done, _, _ = env.step(action)   # Added second _ for new version of gym
            reward *= value
            next_state = np.array(get_state_idxs(next_state), dtype=float)
            next_state[0] = next_state[0]/32
            next_state[1] = next_state[1]/10
        
            agent.step(state, action, reward, next_state, done)   
            state = next_state
            episode_score += reward
        
        scores_window.append(episode_score)
        scores.append(episode_score)
            
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)), end="")
        if episode % 5000 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            
    return scores




def train_agent(env,
                qtable: np.ndarray,
                num_episodes: int,
                alpha: float, 
                gamma: float, 
                epsilon: float, 
                epsilon_decay: float) -> np.ndarray:

    for episode in range(num_episodes):
        state, _ = env.reset()                                     # Added blank for extra returned argument
        done = False
        while True:
            action = get_action(qtable, state, epsilon)
            new_state, reward, done, _, info = env.step(action)    # Added blank for extra returned argument
            qtable = update_qtable(qtable, state, action, reward, new_state, alpha, gamma)
            state = new_state
            if done:
                break
        epsilon = np.exp(-epsilon_decay*episode)
    return qtable


def run_backjack(num_episodes, epsilon, gamma, alpha, decay_rate):
    env = gym.make("Blackjack-v1")
    env.action_space.seed(42)

    # get initial state
    state = env.reset()

    state_size = [x.n for x in env.observation_space]
    action_size = env.action_space.n

    qtable = np.zeros(state_size + [action_size]) #init with zeros


    alpha = alpha # learning rate
    gamma = gamma # discount rate
    epsilon = epsilon    # probability that our agent will explore
    decay_rate = decay_rate

    # training variables
    num_hands = num_episodes

    env = gym.make("Blackjack-v1")
    env.action_space.seed(42)

    agent = Agent(state_size=3, action_size=2, seed=0)
    scores = dqn(n_episodes=70_000)









def parse_args():
    parser = argparse.ArgumentParser()
    os.environ['SM_OUTPUT_DATA_DIR'] = "./output"
    parser.add_argument('--num_episodes', type=int, default=500000)
    parser.add_argument('--epsilon', type=float, default=0.9)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--decay_rate', type=float, default=0.005)
    
    parser.add_argument('--output_data_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    return parser.parse_args()


def main():
    args = parse_args()

    num_episodes = args.num_episodes
    epsilon = args.epsilon
    gamma = args.gamma
    run_backjack(num_episodes=500000, epsilon=0.9, gamma=0.1, alpha=0.3, decay_rate=0.005)
    


if __name__ == '__main__':
    main()