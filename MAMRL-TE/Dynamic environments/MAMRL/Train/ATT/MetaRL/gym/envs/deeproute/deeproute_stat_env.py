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
from MetaRL.gym.envs.deeproute.stat_backend import StatBackEnd


History = 8
action_history = 5

Default_task = {'topo_file': "att.json"}
Demand_file = "att_demand.txt"

# packet size packet latency inflow rate
flow_lambda =[5.0, 0.3]


class DeeprouteStatEnv(gym.Env):
    """
    Define Deeproute environment.

    """

    def __init__(self, task=Default_task):
        super(DeeprouteStatEnv, self).__init__()
        
        self._done = False
        self.max_ticks = 500
        self._task = task
        nodes, edges, demands = read_input(task['topo_file'], Demand_file)
        self.flow_lambda = flow_lambda
        self.backend = StatBackEnd(flow_lambda=self.flow_lambda, links=edges, nodes=nodes, demands=demands, history=History, seed=100)
        self._reward_local = [0] * len(self.backend.nodes)
        self._reward_local_p = [0] * len(self.backend.nodes)
        # action: next hop of current packet at each node
        actions_space = []
        for node in self.backend.nodes:
            action_space = len(self.backend.nodes_connected_links[node.name]) 
            actions_space.append(spaces.Discrete(action_space))
        self.action_space = spaces.Tuple(actions_space)

        observations_space = []
        for _ in self.backend.nodes:
            observation_num = 1 + 1 + action_history
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

        topology_files = ["att.json", "att_task1.json", "att_task2.json", "att_task3.json"]
        task_files = np.random.choice(topology_files, num_tasks, replace=True)
        tasks = [{'topo_file': file} for file in task_files]
        return tasks
        
    def step(self, actions):

        self.take_actions(actions)
        ob = self.get_state()
        
        if self.ticks >= self.max_ticks:
            self._done = True
        
        if self._done:
            rewards = self.get_reward()
        else:
            rewards = [0] * (len(self.backend.nodes) + 1)

        for index in range(len(rewards) - 1):
            rewards[index] -= self.backend.packet_loss_local[index]
            
        return ob, rewards, self._done, self._task

    def take_actions(self, actions):
        
        self.backend.take_actions(actions)
        self.update_estimation()
        self.ticks += 1
        
    def get_packet_loss_and_delivery_time(self):
        packet_loss = self.backend.packet_loss

        average_delivery_time = self.backend._delivery_time / self.backend._delivered_packets

        return packet_loss, average_delivery_time

    def update_estimation(self):
        
        # get local rewards
        
        local_rewards = [0] * len(self.backend.nodes)
        for index, node in enumerate(self.backend.nodes):
       
            if self.backend._delivered_packets_local[index] > 0:

                temp = - self.backend._delivery_time_local[index] / self.backend._delivered_packets_local[index]
            else:
                temp = 0
            local_rewards[index] = temp
            
        for index, local in enumerate(self._reward_local_p):
            
            self._reward_local[index] = local_rewards[index] - local
            
        for index in range(len(self._reward_local_p)):
            temp = 0
            for index1 in self.backend.nodes_connected_nodes[index]:
                temp -= self._reward_local[index1]
            temp = temp / len(self.backend.nodes_connected_nodes[index])
            temp += self._reward_local[index]
            temp = 0.6 * temp
            self._reward_local_p[index] += temp

    def cal_rewards_link(self, actions):

        rewards_for_link = [0] * len(self.backend.nodes)

        for index1, node in enumerate(self.backend.nodes):
            index2 = 0
            for (link, _) in self.backend.nodes_connected_links[node.name]:
                link_avail = self.backend.links_avail[link.name]
                avail_rate = link_avail / link.bw
                if 1 - avail_rate <= 1e-15 and actions[index1] == index2:
                    rewards_for_link[index1] = 1
                index2 += 1

        return rewards_for_link

    def get_reward(self):
        
        rewards = []
        rewards.extend(self._reward_local)

        temp = sum(rewards) / len(rewards)
        rewards.append(temp)
        return rewards
        
    def reset(self):

        self._done = False
        self.ticks = 0
        _, edges, demands = read_input(self._task['topo_file'], Demand_file)

        self.backend.reset(edges, demands)
        
        self._reward_local = [0] * len(self.backend.nodes)
        self._reward_local_p = [0] * len(self.backend.nodes)
 
        return self.get_state()

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
                local_ob.append(self.backend.nodes.index(dst))
            else:
                local_ob.append(-1)
            
            _, max_queue_node = self.backend.nodes_connected_links[node.name][0]

            for to_link, to_node_name in self.backend.nodes_connected_links[node.name]:
                if len(self.backend.nodes_queues[to_node_name]) > len(self.backend.nodes_queues[max_queue_node]):
                    max_queue_node = to_node_name

            for index, node in enumerate(self.backend.nodes):
                if node.name == max_queue_node:
                    local_ob.append(index)
                    break
   
            local_ob.extend(self.backend.nodes_actions_history[node.name][-action_history:])

            ob.append(local_ob)

        return ob
        
    def seed(self, seed):

        random.seed(seed)
        np.random.seed

    def cleanup(self):
        
        self.backend.cleanup()
        

def read_input(filename, demand_file):

    path_to_file = os.getcwd() + "/MetaRL/gym/envs/deeproute/"

    with open(path_to_file + filename) as f:

        js_data = json.load(f)
        nodes = js_data['data']['mapTopology']['nodes']
        edges = js_data['data']['mapTopology']['edges']

    demand = pd.read_csv(path_to_file + demand_file, header=None)

    return (nodes, edges, demand)












