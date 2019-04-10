## README 

This project is aimed to solve a multi-agent problem by training a pair of agents in a Tennis playing environment. 

Without any training, the agents in the environment will randomly control its racket to response to the tennis. Our goal is to train both agent so that they will be playing Tennis like a trained player in real world.

## Observation Spaces and Rewards
To help the agent to understand its correlation between situations (states), response (actions) and scores (rewards), we have observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play, until the sum of the average score of each game play to reach 0.5.

## Getting started

1. Clone the code base
https://github.com/phchoi/drlnd-tennis

2. Environment setup
Please follow the document [here]( https://github.com/udacity/deep-reinforcement-learning#dependencies)

3. Download the Unity Environment
    Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    Mac OSX: [click here]https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip
    Windows (32-bit): [click here]https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip
    Windows (64-bit): [click here]https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip

    Then, place the file in the code directory and unzip (or decompress) the file.

4. Run the Tennis.ipynb in jupter notebook and follow the instructions to learn how to use the code to train the agent


