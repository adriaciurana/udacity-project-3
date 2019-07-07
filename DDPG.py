# coding: utf-8
import os
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from code import Agent, Config
from unityagents import UnityEnvironment

# Check selected env
if Config.SELECTED_ENV == 'tennis':
    ENV_PATH = Config.TENNIS_ENV_PATH
    CHECKPOINT_PATH = Config.CHECKPOINT_TENNIS_PATH
    MODEL_PATH = Config.MODEL_TENNIS_PATH

env = UnityEnvironment(file_name=ENV_PATH)

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])
agent = Agent(num_agents=num_agents, state_size=state_size, action_size=action_size, random_seed=10)

# ### 3. Train the Agent with DDPG
def ddpg(n_episodes=5000, t_max=100000):
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations            
        scores_step = np.zeros(num_agents)             
        score = 0
        agent.reset()
        for step in range(t_max):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]     
            next_states = env_info.vector_observations   
            rewards = env_info.rewards                   
            dones = env_info.local_done                  
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            scores_step += rewards
            if np.any(dones):
                break

        # current score
        current_score = np.max(scores_step) # maximum of the best agent

        # current score to deque and score
        scores_deque.append(current_score)
        scores.append(current_score) 

        # Mean score of window
        mean_score = np.mean(scores_deque)

        print("===============================================")
        print("Episode: %d" % (i_episode, )) 
        print("\t- Current Score: %f (+/- %f)" % (current_score, np.std(scores_step)))
        print("\t- 100 Avg Score: %f (+/- %f)" % (mean_score, np.std(scores_deque)))

        if mean_score > 0.55:
            print("Average score of 0.5 achieved")   
            os.makedirs(CHECKPOINT_PATH, exist_ok=True)
            torch.save(agent.actor_local.state_dict(), os.path.join(CHECKPOINT_PATH, 'checkpoint_actor.pth'))
            torch.save(agent.critic_local.state_dict(), os.path.join(CHECKPOINT_PATH, 'checkpoint_critic.pth'))        
            break
    return scores

torch.save(agent.actor_local.state_dict(), os.path.join(CHECKPOINT_PATH, 'checkpoint_actor.pth'))
torch.save(agent.critic_local.state_dict(), os.path.join(CHECKPOINT_PATH, 'checkpoint_critic.pth'))     

if Config.ENABLE_TRAIN:
    scores = ddpg()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

# ### 4. Watch a Smart Agent!
agent.actor_local.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'checkpoint_actor.pth')))
agent.critic_local.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'checkpoint_critic.pth')))

env_info = env.reset(train_mode=False)[brain_name]
for t in range(200):
    states = env_info.vector_observations
    actions = agent.act(states, add_noise=False)
    env_info = env.step(actions)[brain_name]
    dones = env_info.local_done
    if np.any(dones):
        break 

env.close()