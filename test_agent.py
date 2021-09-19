'''module test_agent.py to test the agent in Tennis Environment.'''

import torch
import numpy as np
from ddpg_agent import Agent
from train_agent import get_environment_info
import argparse
import warnings


def get_ddpg_agent(state_size, action_size, num_agents, random_seed=100):
    return Agent(state_size, action_size, num_agents, random_seed=random_seed)


def prepare_agent(agent, agent1_actor, agent1_critic, agent2_actor, agent2_critic):
    agent.actor_local.load_state_dict(torch.load(agent1_actor, map_location='cpu'))
    agent.critic_local.load_state_dict(torch.load(agent1_critic, map_location='cpu'))
    agent.actor_local.load_state_dict(torch.load(agent2_actor, map_location='cpu'))
    agent.critic_local.load_state_dict(torch.load(agent2_critic, map_location='cpu'))
    return agent


def test_agent_episodes(agent, env, num_agents, brain, brain_name, n_episodes=5):
    for i in range(n_episodes):
        env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
        states = env_info.vector_observations                  # get the current state (for each agent)
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)
        while True:
            actions = agent.act(states,i, add_noise= False)
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            scores += env_info.rewards                         # update the score (for each agent)
            states = next_states                               # roll over states to next time step
            if np.any(dones):                                  # exit loop if episode finished
                break
        print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))



def main():


    parser = argparse.ArgumentParser(description='Train MADDPG agent',
                                    usage='python test_agent.py <path to Tennis Env> <episodes>')
    parser.add_argument("location", help='Input location of the Tennis Evn')
    parser.add_argument("agent1_actor", help='location of weights of Actor network of Agent1')
    parser.add_argument("agent1_critic", help='location of weights of Actor network of Agent1')
    parser.add_argument("agent2_actor", help='location of weights of Actor network of Agent1')
    parser.add_argument("agent2_critic", help='location of weights of Actor network of Agent1')
    parser.add_argument("episodes", type=int, help='Number of episodes to test/play game')

    namespace = parser.parse_args()

    location = namespace.location
    episodes = namespace.episodes
    agent1_actor = namespace.agent1_actor
    agent2_actor = namespace.agent2_actor
    agent1_critic = namespace.agent1_critic
    agent2_critic = namespace.agent2_critic

    warnings.filterwarnings("ignore")

    env, env_info, brain_name, brain, state_size, action_size, num_agents = get_environment_info(location)
    ddpg_agent = get_ddpg_agent(state_size, action_size, num_agents, random_seed=100)
    final_agent = prepare_agent(ddpg_agent, agent1_actor, agent1_critic, agent2_actor, agent2_critic)
    print('DDPG agents have been successfully loaded')

    test_agent_episodes(final_agent, env, num_agents, brain, brain_name, n_episodes=episodes)


if __name__ == '__main__':
    main()