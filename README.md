# Collaboration and Competition - Udacity Deep Reinforcement Learning
Implementation of "Collaboration and Competition" multi-agent reinforcement learning project from Udacity Deep RL Nanodegree.

This is a Deep Reinforcement Learning algorithm to solve the "Tennis"" Unity environment, where the aim is 
to train two agents to collaborate to play tennis.

This repo contains a solution to the environment using the MADDPG algorithm for multi-agent RL.

All the code is contained in the `./src` directory, with `./src/maddpg` for the MADDPG implementation. This 
can be run from the `src/main.py` script (see **"Run"** section below for details).

### Environment
The size of the state space is 24 for each agent, so 24x2, which represents each agents observations .

The size of the action space is 2 for each agent, so 2x2, which represents the lateral and vertical movement for each agent.

The episode ends when the agents fail to keep the ball aloft. The score for an episode is calculated as the maximum score across
both agents.

The environment is considered solved when the average score over the most recent 100 episodes reaches a threshold of 0.5. 
When this threshold is reached the training automatically halts. This threshold can be changed by passing a value to 
the `target_average_score` argument of the MADDPG class in `src/main.py`.

### Getting Started

#### Install basic requirements 
Setup Python environment according to [these instructions](https://github.com/udacity/deep-reinforcement-learning#dependencies) 
in the Udacity [deep-reinforcement-learning](https://github.com/udacity/deep-reinforcement-learning) repo.
eg.
```
conda activate drlnd
pip3 install -r requirements.txt
```

#### Install OpenAI gym
This is required since the Unity environment is wrapped as a gym environment.
```
pip3 install gym
```

#### Download Unity packaged environment
Download the version of the Unity environment for your operating system and move it to the top level of the repo. 

- Linux: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

You will also need to edit line 14 of `src/main.py` to point to the correct filepath of the environment executable for
the `Tennis` environment, eg.
```
env = TennisMultiAgentEnv("./Tennis_Linux/Tennis.x86_64")
```

#### Run
By default the main script loads the saved neural network weights and displays the trained agent acting in the environment. 
So to run in this mode simply run:
```
python src/main.py
```
**OR** to train the neural network weights from scratch pass the `--train` flag to run in training mode:
```
python src/main.py --train
```