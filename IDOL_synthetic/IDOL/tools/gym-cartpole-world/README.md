This repository contains a PIP package which is a modified version of the 
CartPole-v0 OpenAI environment which includes cart & pole friction and random sensor & actuator noise.


## Installation

Install [OpenAI gym](https://gym.openai.com/docs/).

Then install this package via

```
cd gym-cartpole-world
pip install -e .
```

## Usage
Python usage:
```
import gym
import gym_cartpole_world

env = gym.make('CartPoleWorld-v10')
```
Examples:
Versions go from v10 through v19 for different gravity scenarios

## The Environment

Some parameters for the cart-pole system:
- mass of the cart = 1.0
- mass of the pole = 0.1
- length of the pole = 0.5 
- magnitude of the force = 10.0
- friction at the cart = 5e-4
- friction at the pole = 2e-6

Noise cases:
- Noise free
- 5%  Uniform Actuator noise
- 10% Uniform Actuator noise
- 5%  Uniform Sensor noise
- 10% Uniform Sensor noise
- 0.1 var Gaussian Sensor noise
- 0.2 var Gaussian Sensor noise

Note: The sensor noise is added to the angle, theta alone, and the actuator noise is added to the force.





