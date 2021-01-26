#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulate the Deeproute channel selection  environment.

"""

# core modules

import os
import gym
import json
import random
import numpy as np
import pandas as pd
from gym import spaces
from gym_deeproute_stat.envs.stat_backend import StatBackEnd

History = 8
action_history = 5

#
Default_task = {'topo_file': "att.json"}
Demand_file = "att_demand.txt"

# flow size flow latency inflow rate
flow_lambda =[5.0, 0.3]


class DeeprouteStatEnv(gym.Env):
    """
    Define Deeproute environment.

    """

    def __init__(self, task=Default_task):

        self._task = task
        self._done = False
        self.max_ticks = 500
        self.ticks = 0
        nodes, edges, demands = read_input(task['topo_file'], Demand_file)
        self.flow_lambda = flow_lambda
        self.backend = StatBackEnd(flow_lambda=self.flow_lambda, links=edges, nodes=nodes, demands=demands, history=History,
                                   seed=100)
        print(flow_lambda)
        # action: next hop of current packet at each node
        actions_space = []
        for node in self.backend.nodes:
            action_space = len(self.backend.nodes_connected_links[node.name]) 
            actions_space.append(spaces.Discrete(action_space))
        self.action_space = spaces.Tuple(actions_space)

        # Observation: 1) destination of current package in each node 2) queuing length of each node
        observations_space = []
        for _ in self.backend.nodes:
            observation_num = 1 + action_history
            low = np.array([0 for _ in range(observation_num)])
            high = np.array([100 for _ in range(observation_num)])
            observations_space.append(spaces.Box(low, high, dtype=np.float32))
        self.observation_space = spaces.Tuple(observations_space)

    def get_task(self):

        return self._task
        
    def set_task(self, task):
        self._task = task
        self.reset()

    def sample_tasks(self, num_tasks):
        topology_files = ["att.json"]
        topology_file = np.random.choice(topology_files, num_tasks, replace=True)
        tasks = [{'topo_file': file} for file in topology_file]
        return tasks
        
    def step(self, actions):

        self.take_actions(actions)
        reward = self.get_reward()
        ob = self.get_state()
        if self.ticks >= self.max_ticks:
            self._done = True
        # print(self._done)
        flags = []
        for node in self.backend.nodes:
            flags.append(self.backend.flags[node.name])
        return ob, reward, self._done, flags

    def take_actions(self, actions):
        
        self.backend.take_actions(actions)
        self.ticks += 1

    def get_reward(self):
        rewards = []
        for node in self.backend.nodes:
            reward = -self.backend.real_time_reward[node.name]
            rewards.append(reward)
        return rewards
        
    def get_packet_loss_and_delivery_time(self):
        global_packet_loss = self.backend.packet_loss
        global_average_delivery_time = self.backend.delivery_time / self.backend.delivered_packets
        return global_packet_loss, global_average_delivery_time

    def reset(self):
        self._done = False
        self.ticks = 0

        _, edges, demands = read_input(self._task['topo_file'], Demand_file)
        self.backend.reset(edges, demands)
        
        return self.get_state()

    def re_count(self):
        self._done = False
        self.ticks = 0
        self.backend.re_count()

    def render(self):
        
        self.backend.render()

    def get_state(self):
        
        """Get the observation.  it is a tuple """
        ob = []

        # get the destination of current packet
        for node in self.backend.nodes:
            local_ob = []

            if len(self.backend.nodes_queues[node.name]) > 0:
                dst = self.backend.nodes_queues[node.name][0].destination
                local_ob.append(dst.index)
            else:
                local_ob.append(-1)

            ob.append(local_ob)

        return ob

    def seed(self, seed):
        random.seed(seed)
        np.random.seed

    def cleanup(self):
        
        self.backend.cleanup()


def read_input(filename, demand_file):

    path_to_file = os.getcwd() + "/gym_deeproute_stat/envs/"

    with open(path_to_file + filename) as f:

        js_data = json.load(f)
        nodes = js_data['data']['mapTopology']['nodes']
        edges = js_data['data']['mapTopology']['edges']

    demand = pd.read_csv(path_to_file + demand_file, header=None)

    return (nodes, edges, demand)


