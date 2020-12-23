import os
import time
import pylab
import random
import logging
import matplotlib
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors

ETA = 0.7
initial_queue_length = 5

class NODE(object):
    def __init__(self, name, index, posx, posy):
        self.name = name
        self.index = index
        self.pos = (posx, posy)

class LINK(object):
    def __init__(self, name, bw, lat, node1, node2):
        self.name = name
        self.bw = bw
        self.lat = lat 
        self.node2 = node2
        self.node1 = node1

class FlowTraffic(object):
    def __init__(self, bw, dur, destination):
        self.bw = bw
        self.lat = dur
        self.counter = dur
        self.to_link = None
        self.local_lat = dur
        self.to_node_name = None
        self.destination = destination

class StatBackEnd(object):

    def __init__(self, flow_lambda, links, nodes, demands, history, seed):
        np.random.seed(seed)
        self.nodes_queues = {}
        self.active_packets = []
        self._history = history
        self.packet_loss = 0
        self._delivered_packets = 0
        self._generated_packets = 0
        self._delivery_time = 0
        self.nodes_actions_history = {}
        self.flow_lambda = flow_lambda
        self.nodes = self.gen_nodes(nodes)
        self.links = self.gen_edges(links)
        self.demands = self.gen_demands(demands)
        self.ticks = [0] * len(self.nodes)
        self.links_avail = self.gen_links_avail()
        self.packet_loss_local = [0] * len(self.nodes)
        self._delivery_time_local = [0] * len(self.nodes)
        self._delivered_packets_local = [0] * len(self.nodes)
        self.nodes_connected_links, self.nodes_connected_nodes = self.gen_nodes_connected_links()
        self.nodes_Q = self.gen_nodes_Q()

    def gen_nodes_Q(self):
        nodes_Q = {}
        for node1 in self.nodes:
            nodes_Q[node1.name] = {}
            for node2 in self.nodes:
                nodes_Q[node1.name][node2.name] = [0] * len(self.nodes_connected_links[node1.name])

        return nodes_Q

    def gen_demands(self, demands_input):

        Demand_output = {}
        path = 0
        num_nodes = len(self.nodes)
        for index1 in range(num_nodes):
            Demand_output[index1] = {}
            for index2 in range(num_nodes):
                if index1 == index2:
                    continue
                Demand_output[index1][index2] = []
                for index3 in range(len(demands_input[0])):
                    demand = demands_input[0][index3].split()[path]
                    Demand_output[index1][index2].append(float(demand))
                path += 1

        return Demand_output
        
    def gen_nodes_connected_links(self):
        nodes_connected_links = {}
        nodes_connected_nodes = {}
        for index1, node in enumerate(self.nodes):
            nodes_connected_links[node.name] = []
            nodes_connected_nodes[index1] = []
            for link in self.links:
                if link.node1 == node.name or link.node2 == node.name:
                    if link.node1 == node.name:
                        nodes_connected_links[node.name].append((link, link.node2))
                        for index2, connected_node in enumerate(self.nodes):
                            if connected_node.name == link.node2:
                                nodes_connected_nodes[index1].append(index2)
                    else:
                        nodes_connected_links[node.name].append((link, link.node1))
                        for index2, connected_node in enumerate(self.nodes):
                            if connected_node.name == link.node1:
                                nodes_connected_nodes[index1].append(index2)
        return nodes_connected_links, nodes_connected_nodes
                    
    def gen_edges(self, links):
        edgelist = []
        for e in links:
            edge_detail = LINK(e["name"], e["BW"], e["Lat"], e["from"], e["to"])
            edgelist.append(edge_detail)
        return edgelist
        
    def gen_nodes(self, nodes):

        nodeslist = []
        for index, n in enumerate(nodes):
            node_detail = NODE(n["name"], index, n["posx"], n["posy"])
            nodeslist.append(node_detail)
        return nodeslist
        
    def gen_links_avail(self):
        links_avail = {}
        for link in self.links:
            links_avail[link.name] = link.bw
        return links_avail
        
    def generate_queues(self, node_index, node_name, reset=False, K=1, Occur_pro=1):

        if reset:
            self.nodes_queues[node_name] = []

        if np.random.uniform(0,1) <= Occur_pro:
            self.ticks[node_index] = 0
            for _ in range(K):
                self._generated_packets += 1

                new_f_lat = 0
                new_f_destination = np.random.choice(self.nodes, 1, replace=False)
                while new_f_destination[0].name == node_name:
                    new_f_destination = np.random.choice(self.nodes, 1, replace=False)
                new_f_bw = np.random.choice(self.demands[node_index][new_f_destination[0].index], 1, replace=False)[0]
                # if len(self.demands[node_index][new_f_destination[0].index]) > 1:
                #     self.demands[node_index][new_f_destination[0].index].remove(new_f_bw)
                self.nodes_queues[node_name].append(FlowTraffic(new_f_bw, new_f_lat, new_f_destination[0]))
                
    def cleanup(self):
        pass
    
    def reset(self, links, demands):
        self.active_packets.clear()
        self._delivered_packets = 0
        self._generated_packets = 0
        self.packet_loss = 0
        self.demands = self.gen_demands(demands)
        self.packet_loss_local = [0] * len(self.nodes)
        self._delivery_time = 0
        self._delivery_time_local = [0] * len(self.nodes)
        self._delivered_packets_local = [0] * len(self.nodes)
       
        self.links = self.gen_edges(links)
        self.links_avail = self.gen_links_avail()
        
        for index, node in enumerate(self.nodes):
            
            self.generate_queues(index, node.name, reset=True, K=initial_queue_length)
        
        for node in self.nodes:
            if node.name not in self.nodes_actions_history:
                self.nodes_actions_history[node.name] = [] 
            for index in range(self._history):
                new_f_lat = np.random.randint(1, 4)
                new_f_destination = np.random.choice(self.nodes, 1, replace=False)
                while new_f_destination[0].name == node.name:
                    new_f_destination = np.random.choice(self.nodes, 1, replace=False)
                new_f_bw = np.random.choice(self.demands[node.index][new_f_destination[0].index], 1, replace=False)[0]
                # if len(self.demands[node.index][new_f_destination[0].index]) > 1:
                #     self.demands[node.index][new_f_destination[0].index].remove(new_f_bw)
                action = np.random.choice(np.arange(len(self.nodes_connected_links[node.name])), 1)
                self.nodes_actions_history[node.name].append(action[0])
                to_link, to_node_name = self.nodes_connected_links[node.name][action[0]]
                current_packet = FlowTraffic(new_f_bw, new_f_lat, new_f_destination[0])
                current_packet.counter = index + 1
                if self.links_avail[to_link.name] > new_f_bw:
                    self.links_avail[to_link.name] -= new_f_bw
                    current_packet.to_link = to_link
                    current_packet.lat += to_link.lat
                    current_packet.to_node_name = to_node_name
                    self.active_packets.append(current_packet)
            while len(self.nodes_actions_history[node.name]) < self._history:
                self.nodes_actions_history[node.name].append(-1)

    def update_Q(self, node_name, packet_destination_name, packet_lat, to_node_name, action):
        t = min(self.nodes_Q[to_node_name][packet_destination_name])
        temp = ETA * (t + packet_lat - self.nodes_Q[node_name][packet_destination_name][action])

        self.nodes_Q[node_name][packet_destination_name][action] += temp

    def take_actions(self, actions):

        self.packet_loss_local = [0] * len(self.nodes)
        
        for index in range(len(self.ticks)):
            self.ticks[index] += 1
        
        for packet in self.active_packets:
            packet.counter -= 1
        
        for node in self.nodes:
            for packet in self.nodes_queues[node.name]:
                packet.lat += 1
                packet.local_lat += 1
        
        for index, node in enumerate(self.nodes):
            occur_pro = 1 - np.exp(- self.ticks[index] * self.flow_lambda[1]) # 1 - exp(- lambda t)
            # print(occur_pro)
            self.generate_queues(index, node.name, Occur_pro=occur_pro)

        for index, node in enumerate(self.nodes):
            
            if len(self.nodes_queues[node.name]) > 0:
                current_packet = self.nodes_queues[node.name][0]
                to_link, to_node_name = self.nodes_connected_links[node.name][actions[index]] 
                self.nodes_queues[node.name].remove(current_packet)
                self.nodes_actions_history[node.name].append(actions[index])
                if self.links_avail[to_link.name] > current_packet.bw:
                    current_packet.counter += to_link.lat
                    current_packet.lat += to_link.lat
                    current_packet.local_lat += to_link.lat
                    current_packet.to_link = to_link
                    current_packet.to_node_name = to_node_name
                    self.links_avail[to_link.name] -= current_packet.bw
                    self.active_packets.append(current_packet)
                    self.update_Q(node.name, current_packet.destination.name, current_packet.local_lat, to_node_name,
                                  actions[index])
                else:
                    self.packet_loss += 1
                    self.packet_loss_local[node.index] += 1

                    
            while len(self.nodes_actions_history[node.name]) > self._history:
                self.nodes_actions_history[node.name].pop(0)
        
        for packet in self.active_packets:
            if packet.counter <= 0.1:
                self.active_packets.remove(packet)
                if self.links_avail[packet.to_link.name] > 0:
                    self.links_avail[packet.to_link.name] += packet.bw
                if packet.to_node_name != packet.destination.name:
                    packet.local_lat = 0
                    self.nodes_queues[packet.to_node_name].append(packet)
                else:

                    self._delivered_packets += 1
                    self._delivery_time += packet.lat
                    for index, node in enumerate(self.nodes):
                        if packet.to_node_name == node.name:
                            self._delivered_packets_local[index] += 1
                            self._delivery_time_local[index] += packet.lat
                            break

    
    def render(self):
        # Network Figure
        fig = plt.figure()

        G = nx.Graph()
        for node in self.nodes:
            G.add_node(node.name)
            G.nodes[node.name]["pos"] = node.pos
            
        for link in self.links:
            G.add_edge(link.node1,link.node2)

        pos=nx.get_node_attributes(G,'pos')
        # pos = nx.spring_layout(G)
        nodes_labels = {}
        for node in G.nodes():
            nodes_labels[node] = node

        nodes = nx.draw_networkx_nodes(G, pos, node_size=800)
        edges = nx.draw_networkx_edges(G, pos, width=3)
        labels = nx.draw_networkx_labels(G, pos, labels=nodes_labels, font_size=18)
        nx.draw(G,pos)
        plt.savefig('topo.pdf')

        plt.show()
 
 

        


        
        
        
        
        


                    

 
