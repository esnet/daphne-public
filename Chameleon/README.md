<p align="center">
<img src="https://github.com/esnet/daphne-public/tree/master/Chameleon/figires/cham_topo.png" width="100%" height="100%" title="FlexiBO logo">
<p>

# DeepRoute: An AI gym enviroment for Deep Route on Chameleon Testbed (deeproute-gym)

This is a simulation of DEEP-ROUTE link selection experiment. 
DEEP-ROUTE has two links between the University of Chicago(UC) and Texas Advanced Computing Center(TACC) multi-side via Exogeni and Internet2 L2 socket. The experiment is to represent this optimal link selection as a deep reinforcement learning experiment.


# Resources and Tools Required for Reproducibilty
* Note:This experiment uses the Bring-Your-Own-Controller(BYOC) feature provided by Chameleon. For more details onhow to run SDN  experiments with Chameleon go to: https://tinyurl.com/ta79hd9/
 
   * Heat template for controller:    
     Included: ctrl_template.yml
    * Obtain AL2S circuits between UC and TACC and create network.
    * Create a direct stitch circuits between UC and TACC and create network.
    * Create one Compute Node at UC and TACC.
    * Create one Controller Node at TACC using the Ryu Controller.
    * Connect instances to network created above.
    
    
* Bare Metal Nodes:
    * One Client(Compute Node-CHI@UC)
    * One Server (Compute Node-CHI@TACC)
    * Two Cross Traffic•One controller(CHI@TACC)
    * All Compute Nodes - CC-Ubuntu 16.04
    * Controller Node - CC-CentOS7 or CC-Ubuntu 16.04
* Network
    * AL2S by Internet2
    * Two Corsa switches: TACC and UC
    * Direct Stitch Link via Exogeni to connect both sites. For more details on how to stitch go to: https://tinyurl.com/u7yk47p/
* Cross Traffic - Iperf3

```iperf3 -c <server_ip> -t 20 -i 5 -b 100Mbps -u -t 500```
* Controller - RYU OpenFlow

    Included: simple_switch_13_custom_chameleon_org.py
    
# To Reproduce the experiments: 
     
* Login to your Controller node at (TACC) and clone this repo - https://github.com/bashirmohd/rl_chameleon.git

 ```git clone https://github.com/bashirmohd/rl_chameleon.git```
 
    * cd ~/rlchameleon: 
 * Dependencies: install in the following order: ignore errors
 
  ```sudo apt-get install -y python3-setuptools```
  
  ```sudo pytho3 setup.py install```
  
  ```sudo pip3 install -e .```
  
  ```git clone https://github.com/mininet/mininet```
  
  ```cd ~/mininet```
  
  ```git tag```
  
  ```git checkout```
  
  ```cd ..```
  
  ```sudo mininet/util/install.sh -a```
     
  ```pip install gym```
  
  ```sudo pip3 install keras```
  
  ```sudo pip3 install tensorflow```
  
 cd rl_chameleon/deeproute-gym-stat-master
sudo pip3 install -e .
*********
 
* in \tests folder run the following via terminal:
    
        deeproute_rl_dqn_agent.py
        deeproute_rl_random_agent.py
        
* OR the following jupyter notebooks

        deeproute_rl_dqn_agent.ipynb
        deeproute_rl_random_agent.ipynb
        
        Note: After every simulation a .h5 and .csv file is generated in the same directory. 
        This is then used to generate the Performance plot for both Random_Agent and DeepRoute DQN agent. 
        If you get an error message please ==> cd rl_chameleon/deeproute-gym-stat-master ==>run sudo pip3 install -e .
        
        Please note that this experiment was succesfully reproduced using Ubuntu OS.

## Contacts

* Bashir Mohammed
* [Mariam Kiran ](https://sites.google.com/lbl.gov/daphne/home?authuser=0)
        
   
