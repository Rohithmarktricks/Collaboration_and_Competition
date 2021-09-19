import torch
import numpy as np
from ddpg_agent import Agent
from train_agent import get_environment_info
import argparse


def get_ddpg_agent(state_size, action_size, num_agents, random_seed=100):
    return Agent(state_size, action_size, num_agents, random_seed=random_seed)


def prepare_agent(agent, agent1_actor, agent1_critic, agent2_actor, agent2_critic):
    agent.actor_local.load_state_dict(torch.load('checkpoint_agent0_actor.pth', map_location='cpu'))
    agent.critic_local.load_state_dict(torch.load('checkpoint_critic0_critic.pth', map_location='cpu'))
    agent.actor_local.load_state_dict(torch.load('checkpoint_agent1_actor.pth', map_location='cpu'))
    agent.critic_local.load_state_dict(torch.load('checkpoint_critic1_critic.pth', map_location='cpu'))
    return agent


def test_agent_epsidoes(agent, n_episodes=5):
    pass


def main():


    parser = argparse.ArgumentParser(description='Train MADDPG agent',
                                    usage='python test_agent.py <path to Tennis Env> <episodes>')
    parser.add_argument("location", help='Input location of the Tennis Evn')
    parser.add(argument("agent1_actor", help='location of weights of Actor network of Agent1'))
    parser.add(argument("agent1_critic", help='location of weights of Actor network of Agent1'))
    parser.add(argument("agent2_actor", help='location of weights of Actor network of Agent1'))
    parser.add(argument("agent2_critic", help='location of weights of Actor network of Agent1'))
    parser.add_argument("episodes", type=int, help='Number of episodes to test/play game')

    namespace = parser.parse_args()

    location = namespace.location
    episodes = namespace.episodes

    warnings.filterwarnings("ignore")

    env, env_info, brain_name, brain, state_size, action_size, num_agents = get_environment_info(location)
    ddpg_agent = get_ddpg_agent(state_size, action_size, num_agents, random_seed=100)
    final_agent = ddpg_agent(ddpg_agent, agent1_actor, agent1_critic, agent2_actor, agent2_critic)
    print('DDPG agents have been successfully loaded')

    test_agent_episodes(agent, n_episodes=episodes)

