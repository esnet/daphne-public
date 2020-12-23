import os
import gym
import csv

import random
import numpy as np
import cherry as ch
import MetaRL as metaRL
from copy import deepcopy
from policies import Policy

from cherry.algorithms import a2c, trpo
from cherry.models.robotics import LinearValue

import torch
from torch import autograd
from torch.autograd import Variable
from torch.distributions.kl import kl_divergence
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def compute_advantages(baseline, tau, gamma, rewards, dones, states, next_states):

    returns = ch.td.discount(gamma, rewards, dones)
    baseline.fit(states, returns)
    values = baseline(states)
    next_values = baseline(next_states)
    bootstraps = values * (1.0 - dones) + next_values * dones
    next_value = torch.zeros(1, device=values.device)
    return ch.pg.generalized_advantage(tau=tau, gamma=gamma, rewards=rewards, dones=dones, values=bootstraps, next_value=next_value)


def maml_loss(train_episodes, learner, baseline, gamma, tau, device):
    states = train_episodes.state()
    actions = train_episodes.action()
    rewards = train_episodes.reward()
    dones = train_episodes.done()
    next_states = train_episodes.next_state()
    states = states.to(device, non_blocking=True)
    actions = actions.to(device, non_blocking=True)
    rewards = rewards.to(device, non_blocking=True)
    dones = dones.to(device, non_blocking=True)
    next_states = next_states.to(device, non_blocking=True)
    # print(actions)
    log_probs = learner.log_prob(states, actions)
    advantages = compute_advantages(baseline, tau, gamma, rewards,
                                    dones, states, next_states)
    advantages = ch.normalize(advantages).detach()
    return a2c.policy_loss(log_probs, advantages)


def fast_adapt(clone, train_episodes, adapt_lr, baseline, gamma, tau, device):
    loss = maml_loss(train_episodes, clone, baseline, gamma, tau, device)
    gradient = autograd.grad(loss, clone.parameters(), retain_graph=True, create_graph=True)

    return metaRL.algorithms.maml.maml_update(clone, adapt_lr, gradient)


def meta_surrogate_loss(iteration_replays, iteration_policies, policy, baseline, tau, gamma, adapt_lr, device):
    mean_loss = 0.0
    mean_kl = 0.0
    for task_replays, old_policy in zip(iteration_replays, iteration_policies):
        train_replays = task_replays[:-1]
        valid_episodes = task_replays[-1]
        new_policy = metaRL.clone_module(policy)

        # Fast Adapt
        for train_episodes in train_replays:
            new_policy = fast_adapt(new_policy, train_episodes, adapt_lr, baseline, gamma, tau, device)

        # Useful values
        states = valid_episodes.state()
        actions = valid_episodes.action()
        next_states = valid_episodes.next_state()
        rewards = valid_episodes.reward()
        dones = valid_episodes.done()
        
        states = states.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        rewards = rewards.to(device, non_blocking=True)
        dones = dones.to(device, non_blocking=True)
        next_states = next_states.to(device, non_blocking=True)

        # Compute KL
        old_densities = old_policy.density(states)
        new_densities = new_policy.density(states)
        kl = kl_divergence(new_densities, old_densities).mean()
        mean_kl += kl

        # Compute Surrogate Loss
        advantages = compute_advantages(baseline, tau, gamma, rewards, dones, states, next_states)
        advantages = ch.normalize(advantages).detach()
        old_log_probs = old_densities.log_prob(actions).mean(dim=1, keepdim=True)
        new_log_probs = new_densities.log_prob(actions).mean(dim=1, keepdim=True)
        mean_loss += trpo.policy_loss(new_log_probs, old_log_probs, advantages)
        
    mean_kl /= len(iteration_replays)
    mean_loss /= len(iteration_replays)
    
    return mean_loss, mean_kl


def main(env_name,
         adapt_lr, meta_lr, hidden_size, hidden_layers,
         adapt_steps, num_iter, meta_bsz, adapt_bsz,
         tau, gamma, seed, cuda):
             
    print("adapt_lr",adapt_lr)

    cuda = bool(cuda)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device_name = 'cpu'
    if cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        device_name = 'cuda'
    device = torch.device(device_name)
    print(device)

    env = gym.make(env_name)
    
    env.seed(seed)

    '''
    Get the observation size and action size. Now, the global observation space is shared by all the routers, so the observation size is       the same for all the routers. However, each router has a different action size which equals to the number of neighbor routers.
    
    ''' 
    ob_space = env.observation_space
    ob_size = []
    for local_ob in ob_space:
        ob_size.append(local_ob.shape[0])
    action_space = env.action_space
    actions_size = []
    for action in action_space:
        actions_size.append(action.n)
    
    # Each router has an individual policy neural network
    policies = []
    
    for index, act_size in enumerate(actions_size):
        policy = Policy(ob_size[index], act_size, hidden_size, hidden_layers, device=device)
        policy = policy.to(device)
        policies.append(policy)

    # for index in range(len(ob_size)):
    #     file = os.getcwd() + "/savemodels/policy" + "{}".format(index) + ".pth"
    #     policy = torch.load(file)
    #     policy = policy.to(device)
    #     policies.append(policy)
        
    # Current baseline: Benchmarking Deep Reinforcement Learning for Continuous Control
    baselines = []
    for local_ob_size in ob_size:
        baseline = LinearValue(local_ob_size)
        baseline = baseline.to(device)
        baselines.append(baseline)

    # for index in range(len(ob_size)):
    #     file = os.getcwd() + "/savemodels/baseline" + "{}".format(index) + ".pth"
    #     baseline = torch.load(file)
    #     baseline = baseline.to(device)
    #     baselines.append(baseline)
        
    reward_record = []
    global_pl_record = [] 
    global_afct_record = []

    for iteration in range(num_iter):
        iteration_reward = 0.0
        iteration_replays = [[] for _ in range(len(policies))]
        iteration_policies = [[] for _ in range(len(policies))]
   
        for task_topos in env.sample_tasks(meta_bsz):  # Samples batch of tasks
            # clones of the current policies are used to sample trajectories
            clones = []
            for policy in policies:
                clones.append(deepcopy(policy))
            # The topology of the network might change after setting a new task
            env.set_task(task_topos)
            task = metaRL.sample_trajectory(env)
            task_replay = [[] for _ in range(len(policies))]
            # get the initialization of a specific task
            for step in range(adapt_steps):
                train_episodes, global_reward, global_pl, global_afct = task.run(clones, episodes=adapt_bsz)

                for index, clone in enumerate(clones):
                    clone = fast_adapt(clone, train_episodes[index], adapt_lr, baselines[index], gamma, tau, device)
                for index, local_train_episodes in enumerate(train_episodes):
                    task_replay[index].append(local_train_episodes)
                # record
                global_pl_record.append(sum(global_pl) / adapt_bsz)
                global_afct_record.append(sum(global_afct) / adapt_bsz)
            # check if clones change
            valid_episodes, global_reward, packet_loss, completion_time = task.run(clones, episodes=adapt_bsz)
            # record
            global_pl_record.append(sum(global_pl) / adapt_bsz)
            global_afct_record.append(sum(global_afct) / adapt_bsz)
            for index, local_valid_episodes in enumerate(valid_episodes):
                task_replay[index].append(local_valid_episodes)
            
            iteration_reward += sum(global_reward) / adapt_bsz
            for index, clone in enumerate(clones):
                iteration_replays[index].append(task_replay[index])
                iteration_policies[index].append(clone)

        # Print statistics
        print('\nIteration', iteration)
        global_packet_loss, global_average_delivery_time = env.get_packet_loss_and_delivery_time()
        print('packet loss', global_packet_loss)
        print('average_delivery_time', global_average_delivery_time)
        adaptation_reward = iteration_reward / meta_bsz
        reward_record.append(adaptation_reward)
        print('adaptation_reward', adaptation_reward)
        
        if iteration % 1 == 0:
            # save model
            for index, policy in enumerate(policies):
                file = os.getcwd() + "/savemodels/policy" + "{}".format(index) + ".pth"
                torch.save(policy, file)
            for index, baseline in enumerate(baselines):
                file = os.getcwd() + "/savemodels/baseline" + "{}".format(index) + ".pth"
                torch.save(baseline, file)
                
            # save results
            
            with open('results.csv', 'w') as csvfile:
                resultswriter = csv.writer(csvfile, dialect='excel')
                resultswriter.writerow(["name", "value"])
                resultswriter.writerow(["flow", env.flow_lambda])
                resultswriter.writerow(["topology", env.get_task()])
                resultswriter.writerow(["adapt_lr", adapt_lr])
                resultswriter.writerow(["meta_lr", meta_lr])
                resultswriter.writerow(["hidden_size", hidden_size])
                resultswriter.writerow(["hidden_layers", hidden_layers])
                resultswriter.writerow(["meta_bsz", meta_bsz])
                resultswriter.writerow(["adapt_bsz", adapt_bsz])
                resultswriter.writerow(["packet_loss", global_pl_record])
                resultswriter.writerow(["completion_time", global_afct_record])
                resultswriter.writerow(["rewards", reward_record])
            
        # TRPO meta-optimization
        backtrack_factor = 0.5
        ls_max_steps = 30
        max_kl = 0.01
        for index, policy in enumerate(policies):
            # Compute CG step direction
            iter_replay = iteration_replays[index]
            iter_policy = iteration_policies[index]
            baseline = baselines[index]
            old_loss, old_kl = meta_surrogate_loss(iter_replay, iter_policy, policy, baseline, tau, gamma, adapt_lr, device)
            grad = autograd.grad(old_loss, policy.parameters(),retain_graph=True, create_graph=True)
                                 
            grad = parameters_to_vector([g.detach() for g in grad])

            Fvp = trpo.hessian_vector_product(old_kl, policy.parameters())
            step = trpo.conjugate_gradient(Fvp, grad)
            shs = 0.5 * torch.dot(step, Fvp(step))
            lagrange_multiplier = torch.sqrt(shs / max_kl)
            step = step / lagrange_multiplier
            step_ = [torch.zeros_like(p.data) for p in policy.parameters()]
            vector_to_parameters(step, step_)
            step = step_

            del Fvp, grad, old_kl
    
            # print(Steps)   
            old_loss.detach_()
            
            # Line-search
            for ls_step in range(ls_max_steps):
                test_policy = deepcopy(policy)
                stepsize = backtrack_factor ** ls_step * meta_lr
                
                for p, u in zip(test_policy.parameters(), step):
                    p.data.add_(-stepsize, u.data)
                    
                new_loss, new_kl = meta_surrogate_loss(iter_replay, iter_policy, test_policy, baseline, tau, gamma,adapt_lr, device)
    
                if new_loss < old_loss and new_kl < max_kl:
                    for p, u in zip(policy.parameters(), step):
                        p.data.add_(-stepsize, u.data)
                    print("true")
                    break
    # save model
    for index, policy in enumerate(policies):
        file = os.getcwd() + "/savemodels/policy" + "{}".format(index) + ".pth"
        torch.save(policy, file)
    for index, baseline in enumerate(baselines):
        file = os.getcwd() + "/savemodels/baseline" + "{}".format(index) + ".pth"
        torch.save(baseline, file)
    
    # save results
    
    with open('results.csv', 'w') as csvfile:
        resultswriter = csv.writer(csvfile, dialect='excel')
        resultswriter.writerow(["name", "value"])
        resultswriter.writerow(["flow", env.flow_lambda])
        resultswriter.writerow(["topology", env.get_task()])
        resultswriter.writerow(["adapt_lr", adapt_lr])
        resultswriter.writerow(["meta_lr", meta_lr])
        resultswriter.writerow(["hidden_size", hidden_size])
        resultswriter.writerow(["hidden_layers", hidden_layers])
        resultswriter.writerow(["meta_bsz", meta_bsz])
        resultswriter.writerow(["adapt_bsz", adapt_bsz])
        resultswriter.writerow(["packet_loss", global_pl_record])
        resultswriter.writerow(["delivery_time", global_afct_record])
        resultswriter.writerow(["rewards", reward_record])
                    

if __name__ == '__main__':

    main(env_name='Deeproute-stat-v0',
         adapt_lr=0.01,
         meta_lr=1,
         hidden_size=128,
         hidden_layers=5,
         adapt_steps=1,
         num_iter=300,
         meta_bsz=3,
         adapt_bsz=15,
         tau=1.00,
         gamma=1.00,
         seed=42,
         cuda=1)

