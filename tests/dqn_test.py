import gym
import numpy as np
from utils import experience_replay, epsilon_scheduler, constants
from Agents import dqn_agent

def run():
    print("Unit tests for DQN agent")
    print()
    dqn_agent.run_tests()
    print()
    print("Integration test for DQN agent")
    print()

    env = gym.make('ALE/MsPacman-v5')
    observation, _ = env.reset()
    inactive_frames = 65

    config = constants.load

    replay_buffer = experience_replay.ReplayBuffer(100)
    scheduler = epsilon_scheduler.EpsilonScheduler()
    agent = dqn_agent.DQNAgent(env, config, replay_buffer=replay_buffer, scheduler=scheduler)

    for _ in range(inactive_frames):
        state, reward, done, info, _ = env.step(0)

    print("Exploring for 10 steps")
    scheduler.current_epsilon = 1
    for i in range(10):
        action = agent.act(state)
        next_state, reward, done, info, _ = env.step(action)

        replay_buffer.add(state, action, next_state, reward, done)

        state = next_state

        agent.train()

        if done:
            state, _ = env.reset()

    print("Exploiting for 10 steps")
    scheduler.current_epsilon = 0
    for i in range(10):
        action = agent.act(state)
        next_state, reward, done, info, _ = env.step(action)

        replay_buffer.add(state, action, next_state, reward, done)

        state = next_state

        agent.train()

        if done:
            state, _ = env.reset()

    env.close()

    print()
    print("Integration test for DQN agent completed. No errors found.")
    print()

    


