# Continuous Control - Udacity Deep Reinforcement Learning
Implementation of "Continuous Control" project from Udacity Deep Reinforcement Learning Nanodegree.

This is a Deep Reinforcement Learning algorithm to solve the "Continuous Control" Unity environment, where the aim is 
to keep a robot arm with multiple joints within a target sphere.

This project contains a solution to Version 2 of the continuous control environment, with 20 parallel agents.

It contains implementations of both the DDPG and PPO algorithm, however only the DDPG implementation successfully solves
the environment at present.

All the code is contained in the `./src` directory, with `./src/ddpg` for the DDPG implementation and `./src/ppo` for PPO. They 
are both run from the `src/main.py` script, and the choice of algorithm can be configured by passing a command line argument (see **"Run"** section 
below for details).

### Environment
The size of the state space is 33, which represents the agent's velocity and perception of nearby objects.

The size of the action space is 4, which represents the torque applied to each joint.

The environment is considered solved when the agent can keep the end of the robot arm within the target sphere. 
Roughly this corresponds to a score of 30.0, which is calculated from the average reward received by all agents in an episode. 
Once the average score over all agents and over the most recent 100 episodes reaches this threshold 
of 30.0 then it is considered solved and the training stops. This threshold can be changed by passing a value
 to the `target_reward` argument of the DDPG class in `src/main.py`.

### Getting Started

#### Install basic requirements
```
conda activate drlnd
pip3 install -r requirements.txt
```

#### Install OpenAI gym
```
pip3 install gym
```

#### Download Unity packaged environment
Download the version of the Unity environment for your operating system and move it to the top level of the repo. 

**NB** make sure you download _Version 2_ of the environment, with 20 agents. 

You will also need to edit line 18 of `src/main.py` to pass the correct filepath of the environment executable to
the `ReacherMultiAgentEnv` object.

#### Run
By default the main script loads the saved neural network weights and displays the trained agent acting in the environment. So to run
in this mode simply run:
```
python src/main.py
```
**OR** to train the neural network weights from scratch pass the `--train` flag to run in training mode:
```
python src/main.py --train
```

By default the DDPG algorithm is used for training, however you can instead choose to use the PPO algorithm by passing the `--ppo` flag, eg.
```
python src/main.py --train --ppo
```