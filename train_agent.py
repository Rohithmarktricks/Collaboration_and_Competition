import torch
from maddpg import MADDPG
from ddpg_agent import Agent
from collections import deque
import time, os
import numpy as np
import pandas as pd
import argparse
import warnings
from unityagents import UnityEnvironment
from time import strftime

def get_environment_info(location):
    env = UnityEnvironment(file_name=location)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    print('Number of agents: ',num_agents)

    action_size = brain.vector_action_space_size
    print('Size of each action: ', action_size)

    states = env_info.vector_observations
    state_size = states.shape[1]

    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like: ', states[0])

    return env, env_info, brain_name, brain, state_size, action_size, num_agents


def get_agent(state_size, action_size, num_agents, random_seed):
    maddpg = MADDPG(state_size, action_size, num_agents, random_seed)
    # agent = Agent(state_size, action_size, num_agents, random_seed)
    return maddpg


def maddpg_train(env, env_info, brain_name, brain, num_agents, maddpg, train_mode=True, n_episodes=3000):

    scores_max_hist = []
    scores_mean_hist = []

    scores_deque = deque(maxlen=100)
    solved = False

    for i_episode in range(n_episodes):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        scores = np.zeros(num_agents)
        maddpg.reset()
        step = 0
        while not solved:
            step += 1
            action = maddpg.act(state, i_episode, add_noise=True)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations
            reward = env_info.rewards
            done = env_info.local_done

            scores += reward

            maddpg.step(i_episode, state, action, reward, next_state, done)

            if np.any(done):
                break

            state = next_state

        score_max = np.max(scores)
        scores_deque.append(score_max)
        score_mean = np.mean(scores_deque)
        scores_max_hist.append(score_max)
        scores_mean_hist.append(score_mean)

        print('\r{} episode\tavg score {:.5f}\tmax score {:.5f}'.format(i_episode, np.mean(scores_deque), score_max), end='')
        if solved == False and score_mean > 0.5:
            start_time = strftime("%Y%m%d-%H%M%S")
            print('\nEnvironment solved after {} episodes with the average score {}\n'.format(i_episode, score_mean))
            maddpg.save(start_time)
            print('saved the models weights in models folder..')
            max_scores_fname = f"scores/maddpg_agent_max_scores_{start_time}.csv"
            np.savetxt(max_scores_fname, scores_max_hist, delimiter=',')
            mean_scores_fname = f"scores/maddpg_agent_mean_scores_{start_time}.csv"
            np.savetxt(mean_scores_fname, scores_mean_hist, delimiter=',')

            print('saved the max and mean scores of the agents in scores folder.')
            solved = True
        
        if i_episode % 500 == 0:
            print()


def main():
    parser = argparse.ArgumentParser(description='Train MADDPG agent',
                                    usage='python train_agent.py <path to Tennis Env> <episodes>')
    parser.add_argument("location", help='Input location of the Tennis Evn')
    parser.add_argument("episodes", type=int, help='Number of episodes to train agent')

    namespace = parser.parse_args()

    location = namespace.location
    episodes = namespace.episodes

    warnings.filterwarnings("ignore")

    env, env_info, brain_name, brain, state_size, action_size, num_agents = get_environment_info(location)
    maddpg = get_agent(state_size, action_size, num_agents, random_seed=100)
    print("MADDPG agents have been successfully loaded!")
    print()
    print(f"Training the agents for {episodes} episodes")
    maddpg_train(env, env_info, brain_name, brain, num_agents, maddpg, train_mode=True, n_episodes=episodes)

if __name__ == "__main__":
    main()