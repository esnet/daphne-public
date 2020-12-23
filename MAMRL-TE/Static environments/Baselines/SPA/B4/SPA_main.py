#!/usr/bin/env python
# coding: utf-8

import random
import gym
import sys
import networkx as nx
import gym_deeproute_stat
from Dijkstra import dijkstra
import matplotlib.pyplot as plt
import csv
import numpy as np

num_iteration = 3000
iteration = 0
spa_delivery_time_record = []
spa_packet_loss_record = []

env = gym.make('Deeproute-stat-v0')

MAX_TICKS = env.max_ticks

# SPA
observation = env.reset()
for task in env.sample_tasks(num_iteration):
    print(task)
    print("iteration", iteration)
    iteration += 1
    observation = env.reset()
    # env.set_task(task)
    env.re_count()
    done = False
    while not done:
        actions = []
        for node in env.backend.nodes:
            if len(env.backend.nodes_queues[node.name]) > 0:
                flow = env.backend.nodes_queues[node.name][0]
                nodes = env.backend.nodes
                nodes_connected_links = env.backend.nodes_connected_links
                action = dijkstra(nodes, nodes_connected_links, node.name, flow.destination.name)
                actions.append(action)
            else:
                actions.append(0)

        observation, reward, done, task = env.step(actions)
    global_packet_loss, global_average_delivery_time = env.get_packet_loss_and_delivery_time()
    spa_delivery_time_record.append(global_average_delivery_time)
    spa_packet_loss_record.append(global_packet_loss)
    print('packet loss', global_packet_loss)
    print('average_delivery_time', global_average_delivery_time)
    print("generated_packets:", env.backend.generated_packets)
    print("delivered_packets:", env.backend.delivered_packets)


with open('results.csv', 'w') as csvfile:
    resultswriter = csv.writer(csvfile, dialect='excel')
    resultswriter.writerow(["name", "value"])
    resultswriter.writerow(["flow", env.flow_lambda])
    resultswriter.writerow(["topology", env.get_task()])
    resultswriter.writerow(["packet_loss", spa_packet_loss_record])
    resultswriter.writerow(["completion_time", spa_delivery_time_record])

fig, ax = plt.subplots()
ax.plot(spa_delivery_time_record, 'r', label='SPA')
ax.set_ylabel("Average packet completion time")
ax.set_xlabel("Iterations")
plt.title("inflow rate = {}".format(env.backend.flow_lambda[1]))
leg = ax.legend()
plt.grid()

fig, ax = plt.subplots()
ax.plot(spa_packet_loss_record, 'r', label='SPA')
ax.set_ylabel("packet loss")
ax.set_xlabel("Iterations")
plt.title("inflow rate = {}".format(env.backend.flow_lambda[1]))
leg = ax.legend()


plt.grid()
plt.show()

