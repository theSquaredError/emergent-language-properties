""" ##Hyperparameters
* location_space: the number of vertexes in the 2D world
* n_sectors: numbers of sectors for the agent
* n_segments: number of segments for the agent
* n_colors: number of colors for the agent

"""
from utils import DotDic
from pprint import pprint
from environment import Environment
from policy_network import SNet, LNet
import itertools
from train import train
import torch
from graphworld import World
import sys
import utils
# concept space is ordered as follows: [sector[0-7], segment[8-11], color[12-15]]

sys.stdout = open("console_output.txt", "w")
order_vec = list(itertools.permutations(['segment', 'sector', 'color']))
opts={
    'n_agents':2,
    'n_vertex': 2,
    
    'n_sectors': 4, 
    'n_segments': 4, 
    'n_colors': 2,
    'n_concepts':10,
    'n_vocab': 3,
    'obv_vec_size':13,
    'order_vec': order_vec,
    'order_vec_size' : len(order_vec),
    'max_DIMENSIONALITY': 10,
    'min_DIMENSIONALITY': -10,
    'RGB_value' : [[0.6, 0.0, 0.3],
                [0.2, 0.2, 1.0]],
    'positve_reward': 1,
    'negative_reward': -1,


    'rnn_hidden_size': 128,

    'rnn_size':128,
    "lr" : 0.0005,
    "momentum" : 0.05,
    "eps":0.05,
    "nepisodes":500,
}
radiuses = torch.linspace(0, 20, steps= opts['n_segments'])
opts['radiuses'] = radiuses

world = World(DotDic(opts))

utils.saveGraph(world=world, opts=DotDic(opts))

# pprint(opts)

# create two policy 
speaker_policy = SNet(opts  = DotDic(opts))
listener_policy = LNet(opts = DotDic(opts))


env = Environment(DotDic(opts), world)


train(env, speaker = speaker_policy, listener = listener_policy, opts = DotDic(opts), world = world)

sys.stdout.close()