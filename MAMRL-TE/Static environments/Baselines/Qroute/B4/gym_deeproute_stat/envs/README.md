# Deeproute Gym environment



## Input
The input of the deeproute gym environment is a xxx.json flie, where nodes and edges are defined. For example,
~~~bash
nodes: {
          "name": "A",       # the name of the node
          "posx": 0.25,      # the position of the node
          "posy": 0.75
        }
        
edges: {
          "name":"edge5",    # the name of the edge
          "from":"C",        # the two nodes that this edge is connected with
          "to":"F",          # the edge is undirected, so there are no differences between "from" node and "to" node
          "BW":50,           # the bandwidth of this edge
          "Lat":6.           # the latency of this edge
          }
~~~


## Flow models

~~~
Flow size: poisson distribution **lambda = 5**
Flow occurance probability at each step: **1 - exp(-lambda*t)**. Here, t denotes the **time index** and **lambda = 0.5**.  

Notes: Each flow has a source node and a destination node. The flows are travelling in the network, and if a flow has arrived at the destination node, this flow will disappear from the network environment. 
~~~

## Key functions
actions is list of integers, where the ith element of the action of the ith node. The order of the nodes in the gym environment is the same as the order of nodes in the json file. 

observation is is array. 

reward is single number. 


Actions are randomly selected:
~~~
env.reset()

1)generate **3** waiting flow at each node
2)generate ** 8 ** active flows at each link
~~~




~~~
env.set_task(task) 

set the bandwidth of the disappeared link to zero.
~~~


~~~
env.step(actions) 

1) Generate a new waiting flow at each node based on the flow occurance probability. For example, if the flow occurance probability is 80% at one node, then a new waiting flow will be generated at this node with the probability of 80%. 
2) Send out one flow at each node based on the **first in first out** criterion. 
3) Check all the active flows. Here, the active flows means all the flows travelling in the links instead of waiting in the nodes. If a flow has arrived at a node, add this flow to the flow queue of this node. If a flow has arrived at its destination node, remove this flow from the network.
~~~

