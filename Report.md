# Report of Tennis project

This project is one of the assignment of Udacity Deep Reinforcement Learning Nanodegree. The description of the environment can be found [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis). In ths environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play. The goal of this project is to achieve average score of 0.5 or above in 100 episode.

## State and Action Spaces
The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. That all sum up to observation spaces of 24 and 2 actions.

## Learning Algorithm
My implementation of this project is based on the MADDPG (Multi-Agent Actor-Critic)[https://arxiv.org/pdf/1706.02275.pdf] architecture with DDPG (Deep Deterministic Policy Gradient) architecture. In this implementation, a combination of Multi-Agent actor-critic approach and memory replay are used for the training.

Because this is a tennis game and it has multi-agent, by default each agent will only return its own state but not the other. However, as I learn from the MADDPG paper, training the agent with the opponent states will help, I merged the states from the agent itself with the opponent's agent states. That ends up with a larger input to the neural network and I train the network with the merged states. 

The network of actor and critics are slightly different. The actor network has input of state sizes 24, which hidden layers of 400 and 300 units respectively and a output layer of 2 actions. The critic network has input of action sizes as 52, which is the sum of 2 agents' states size plus its action sizes (24x2+2x2). 

The hyperparameter I used are listed below as well, turns out the size of batch size and TAU (soft update of target parameters) played a major role in improving the training time.

Replay buffer size: 100000  
Minibatch size: 512  
Discount facotr: 0.99  
Soft update of target parameters: 0.1  
Learning rate of the actor: 0.0001  
Learning rate of the critic: 0.0001  
L2 weight decay: 0  

## Results

Episode 100	Average Score: -0.00400 Lowest Score: -0.01000 Highest Score: 0.10000 Elapsed time: 62 secs  
Episode 200	Average Score: -0.00400 Lowest Score: -0.01000 Highest Score: 0.10000 Elapsed time: 151 secs  
Episode 300	Average Score: 0.02900 Lowest Score: -0.01000 Highest Score: 0.20000 Elapsed time: 317 secs  
Episode 400	Average Score: 0.05595 Lowest Score: -0.01000 Highest Score: 0.20000 Elapsed time: 539 secs  
Episode 500	Average Score: 0.05100 Lowest Score: -0.01000 Highest Score: 0.29000 Elapsed time: 754 secs  
Episode 600	Average Score: 0.04750 Lowest Score: -0.01000 Highest Score: 0.20000 Elapsed time: 961 secs  
Episode 700	Average Score: 0.05695 Lowest Score: -0.02000 Highest Score: 0.30000 Elapsed time: 1200 secs  
Episode 800	Average Score: 0.06800 Lowest Score: -0.01000 Highest Score: 0.50000 Elapsed time: 1466 secs  
Episode 900	Average Score: 0.06300 Lowest Score: -0.01000 Highest Score: 0.40000 Elapsed time: 1718 secs  
Episode 1000	Average Score: 0.06695 Lowest Score: -0.01000 Highest Score: 0.30000 Elapsed time: 1976 secs  
Episode 1100	Average Score: 0.06850 Lowest Score: -0.01000 Highest Score: 0.40000 Elapsed time: 2249 secs  
Episode 1200	Average Score: 0.07500 Lowest Score: -0.01000 Highest Score: 0.30000 Elapsed time: 2535 secs  
Episode 1300	Average Score: 0.06295 Lowest Score: -0.01000 Highest Score: 0.30000 Elapsed time: 2800 secs  
Episode 1400	Average Score: 0.08900 Lowest Score: -0.01000 Highest Score: 0.50000 Elapsed time: 3117 secs  
Episode 1500	Average Score: 0.09700 Lowest Score: -0.01000 Highest Score: 0.50000 Elapsed time: 3452 secs
Episode 1600	Average Score: 0.09500 Lowest Score: -0.01000 Highest Score: 0.50000 Elapsed time: 3781 secs  
Episode 1700	Average Score: 0.14050 Lowest Score: -0.01000 Highest Score: 0.80000 Elapsed time: 4226 secs  
Episode 1800	Average Score: 0.12000 Lowest Score: -0.01000 Highest Score: 1.00000 Elapsed time: 4628 secs  
Episode 1900	Average Score: 0.13650 Lowest Score: -0.01000 Highest Score: 0.90000 Elapsed time: 5077 secs  
Episode 2000	Average Score: 0.09500 Lowest Score: -0.01000 Highest Score: 0.60000 Elapsed time: 5415 secs  
Episode 2100	Average Score: 0.18940 Lowest Score: -0.01000 Highest Score: 1.30000 Elapsed time: 5993 secs  
Episode 2200	Average Score: 0.38600 Lowest Score: -0.01000 Highest Score: 2.20000 Elapsed time: 7088 secs  
Episode 2250	Average Score: 0.50560 Lowest Score: -0.01000 Highest Score: 2.60000 Elapsed time: 7921 secs  

## Idea of future works
Definitely would be try adding optimization methods like Proximal Policy Optimization (PPO) and see will it be able to show any impact to training behavior and performance. 
