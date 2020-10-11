# Report: Collaboration and Competition project

This project contains a solution the multi-agent "Tennis" environment, with 2 agents that are both collaborating and 
competing in order to rally a tennis ball back and forth.

I was able to solve this problem using MADDPG, the implementation is described below.

## Implementation
The key to MADDPG is that information from the policies of _all_ agents is used to update each agent, that way the agents
training can take into account the predicted actions of other actions (this prevents problems with non-stationarity that
occur with multi-agent reinforcement learning). In order to allow this the code is structured with a top level `MADDPG` class 
(in `src/main/maddpg.py`) which contains both individual agents, which are instances of the `DDPG` class, because they are essentially
just DDPG agents themselves. Each DDPG agent has both actor and critic networks, with target and local instances.

### Update function


Some specific notable features are... 

Exploration noise...

I experimented with adding noise to the parameter weights, however it seemed like this harmed training performance, so this is now
turned off

### Wrapping as a gym environment
The unity environment was wrapped as an OpenAI gym environment, which encapsulates the unity specific code and results in tidier
code in the actual algorithm, as well as better portability between algorithms. Specifically it was wrapped as a gym `VectorEnv` environment, 
which is an extension of the normal `gym.Env` designed for parallel environments that represent multiple environments at once, eg. they take a stacked vector of actions and 
return a stacked vector of observations and rewards. This obviously fits well with this Tennis Unity environment, since it runs both agents at once and 
returns corresponding vectors.

#### Reward scaling
I fuond that multiplying the rewards returned from the environment by a constant factor of 10.0 improved training performance. 
To account for this when reporting the score for each episode I divided the score for the episode by a factor of 10.0, 
which should result in an equivalent score as if receiving the unscaled rewards from the environment.



#### Neural Net Architecture
The neural net architectures for both actor and critic, as well as the training hyperparameters, were similar
to the example implementations of DDPG. However I used `tanh` activations everywhere instead of `ReLU`.

##### Actor
The actor network is a simple MLP (multi-layer perceptron) architecture with two hidden linear layers of size 400 and 300.
The actor network uses the default PyTorch initialisation for the values of the weights, rather than explicitly initialising
the weights, this was found to improve performance.

It uses `tanh` activations after every layer, including the output (which conveniently keeps the actions between -1 and 1).

##### Critic
The critic network is also a simple MLP architecture but with three hidden layers, the first layer is size 400, the second
layer is a concatenation of the first layer output and the action values, and the third layer has 300 linear units.
The critic network does explicitly initialise the weights, by setting them to a uniform sample scaled by the size of the layer.

It uses `tanh` activations after every layer, apart from the output layer which has no activation function.

## Training 
The training was stopped when the average reward for the last 100 episodes was greater than the 
`reward_threshold` of 30.0. When the average score achieves this threshold the 
training is automatically halted, and the final weights of the neural networks are written to a file. 
Additionally the current weights of the local network were periodically stored to a checkpoint file, in order to 
have a record of the weights during training. 

The "Adam" optimizer was used to update the parameters for both neural networks (actor and critic), however with separate values of learning rate
and L2 weight decay (see hyperparameters section for details).


#### Hyper-parameters
The following table lists all the relevant hyperparameters for the DDPG implementation. The values below were found to give good 
performance, and these were used for the final run which generated the results in the next section (the score plot and gif):

Hyperparameter | Description | Value
--- | --- | ---
gamma | Discount factor | 0.99 
network_update_period | how often to update actor and critic networks (in timesteps) | 20
num_network_updates | how many times to update actor and critic networks | 10
buffer_size | replay buffer size | 1e6
batch_size | minibatch size | 128 
tau | rate of mixing for soft update of target parameters | 1e-3
learning_rate_actor | ADAM learning rate for actor network | 1e-4 
learning_rate_critic | ADAM learning rate for critic network | 2e-4 
critic_weight_decay | L2 weight decay for critic network | 0.0001  


### Results
The target score of 0.5, averaged over 100 timesteps, was reached in **3506 episodes**. The score seemed to increase
fairly consistently, if quite slowly, over the course of the training, and seemed to be on an upward trajectory when the 
training was automatically halted because it had reached the average score target of 0.5. I'm confident that with more 
hyperparameter tuning this agent could reach good performance much faster.

Subjectively the performance on the task is still not great, the agents are able to achieve rallies of a few hits back and forth,
but they still do not seem to have attained natural looking good performance on the task. However given that the average score
was still improving at the end of the training run, this gives confidence that better performance could be achieved by 
training for longer or with better hyperparameter tuning.

![Score by episode](img/reached_target.png "Score")

### Ideas for future work
This project achieved good performance and fast training using DDPG, however there are many further avenues for exploration.
I was surprised that adding noise seemed to hurt training performance, so it would definitely be worthwhile to investigate this
further and see if adding noise could be used to speed up training. 

I didn't spend that much time tuning the hyperparameters for DDPG, as I spent most of that time trying to tune the hyperparameters
for PPO to get it to work, so it would be useful to try even more hyperparameter options for DDPG, perhaps using a
black box optimisation algorithm like bayesian optimisation to automate this process.

There is plenty of things to try out with the neural networks for DDPG, for example experimenting with different sizes and number of
layers. It would also be interesting to perform some more rigorous experiments on whether `tanh` or `ReLu` activations perform best.

It would be good to investigate why the PPO implementation is not working, I already spent quite a lot of time debugging it
but I couldn't get to the bottom of the issue, it's perhaps a simple bug somewhere.

Also now that the environment is wrapped as an OpenAI gym object, it would be relatively easy to try this out with many other RL algorithms
in order to see which ones perform best, for example A3C or A2C, of which there are many open source implementations online which conform
to the gym interface. 