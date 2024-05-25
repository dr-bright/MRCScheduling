
print('Importing')
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from utils import SchedulingEnv, build_hetgraph, hetgraph_node_helper
from benchmark.edfutils import RobotTeam
from graph.hetgat import HeteroGATLayer


torch_dev = 'cpu'


# input similar to HeteroGATLayer
# merge = 'cat' or 'avg'
class MultiHeteroGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, cetypes,
                 num_heads, merge='cat'):
        super(MultiHeteroGATLayer, self).__init__()
        
        self.num_heads = num_heads
        self.merge = merge
        
        self.heads = nn.ModuleList()
        
        if self.merge == 'cat':        
            for i in range(self.num_heads):
                self.heads.append(HeteroGATLayer(in_dim, out_dim, cetypes))
        else:
            #self.relu = nn.ReLU()
            for i in range(self.num_heads):
                self.heads.append(HeteroGATLayer(in_dim, out_dim, cetypes,
                                                 use_relu = False))            

    def forward(self, g, feat_dict):
        tmp = {}
        for ntype in feat_dict:
            tmp[ntype] = []
            
        for i in range(self.num_heads):
            head_out = self.heads[i](g, feat_dict)
            
            for ntype in feat_dict:
                tmp[ntype].append(head_out[ntype])
        
        results = {}
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)  
            for ntype in feat_dict:
                results[ntype] = torch.cat(tmp[ntype], dim=1)
        else:
            # merge using average
            for ntype in feat_dict:
                # dont use relu as the predicted q scores are negative
                results[ntype] = torch.mean(torch.stack(tmp[ntype]), dim=0)
        
        return results


class ScheduleNet4Layer(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, cetypes, num_heads=4):
        super(ScheduleNet4Layer, self).__init__()
        
        hid_dim_input = {}
        for key in hid_dim:
            hid_dim_input[key] = hid_dim[key] * num_heads
        
        self.layer1 = MultiHeteroGATLayer(in_dim, hid_dim, cetypes,
                                          num_heads)
        self.layer2 = MultiHeteroGATLayer(hid_dim_input, hid_dim, cetypes, 
                                          num_heads)
        self.layer3 = MultiHeteroGATLayer(hid_dim_input, hid_dim, cetypes, 
                                          num_heads)
        self.layer4 = MultiHeteroGATLayer(hid_dim_input, out_dim, cetypes, 
                                          num_heads, merge='avg')
    
    '''
    input
        g: DGL heterograph
            number of Q score nodes = number of available actions
        feat_dict: dictionary of input features
    '''
    def forward(self, g, feat_dict):
        h1 = self.layer1(g, feat_dict)
        h2 = self.layer2(g, h1)
        h3 = self.layer3(g, h2)
        h4 = self.layer4(g, h3)
        
        return h4


def load_model(checkpoint_tar):
    # device = torch.device('cuda')
    # use this for CPU-only mode
    device = torch.device('cpu')

    in_dim = {'task': 6,
            'loc': 1,
            'robot': 1,
            'state': 4,
            'value': 1
            }

    hid_dim = {'task': 64,
            'loc': 64,
            'robot': 64,
            'state': 64,
            'value': 64
            }

    out_dim = {'task': 32,
            'loc': 32,
            'robot': 32,
            'state': 32,
            'value': 1
            }

    cetypes = [('task', 'temporal', 'task'),
            ('task', 'located_in', 'loc'),('loc', 'near', 'loc'),
            ('task', 'assigned_to', 'robot'), ('robot', 'com', 'robot'),
            ('task', 'tin', 'state'), ('loc', 'lin', 'state'), 
            ('robot', 'rin', 'state'), ('state', 'sin', 'state'), 
            ('task', 'tto', 'value'), ('robot', 'rto', 'value'), 
            ('state', 'sto', 'value'), ('value', 'vto', 'value'),
            ('task', 'take_time', 'robot'), ('robot', 'use_time', 'task')]

    num_heads = 8
    model = ScheduleNet4Layer(in_dim, hid_dim, out_dim,
                              cetypes, num_heads).to(device)
    cp = torch.load(checkpoint_tar, map_location=device)
    model.load_state_dict(cp['policy_net_state_dict'])
    return model


def load_task(task_path):
    env = SchedulingEnv(task_path)
    robots = RobotTeam(env.num_robots)
    return env, robots


def gnn_pick_task(hetg, act_task, pnet, ft_dict):
    '''
    Pick a task using GNN value function
        hetg: HetGraph in DGL
        act_task: unscheduled/available tasks
        pnet: trained GNN model
        ft_dict: input feature dict
        rj: robot chosen, not needed as hetg is based on the selected robot
        + also returns predictions of each task
    '''
    length = len(act_task)
    if length == 0:
        return -1, np.array([0])
       
    '''
    pick task using GNN
    '''
    #idx = np.argmin(tmp)
    if length == 1:
        idx = 0
        q_s_a_np = np.array([1])
    else:
        with torch.no_grad():
            result = pnet(hetg, ft_dict)
            # Lx1
            q_s_a = result['value']
            q_s_a_np = q_s_a[:,0].data.cpu().numpy()
            # get argmax on selected robot
            a_idx = q_s_a.argmax()
            idx = int(a_idx)
    
    task_chosen = act_task[idx]

    return task_chosen, q_s_a_np


def schedule(task_path, policy_net, map_width):
    env, robots = load_task(task_path)
    # parameters for logging the solving process
    terminate = False
    feas_count = 0
    decision_step = 0

    for t in range(env.num_tasks * 10):
        exclude = []
        rob_chosen = robots.pick_robot_by_min_dur(t, env, 'v1', exclude)
        # Repeatedly select robot with min duration until none available
        while rob_chosen is not None:
            unsch_tasks = np.array(env.get_unscheduled_tasks(),
                                   dtype=np.int64)
            valid_tasks = np.array(env.get_valid_tasks(t),
                                   dtype=np.int64)
            
            if len(valid_tasks) > 0:
                # TODO: study hetgraph construction
                # TODO: reimplement build_hetgraph to support non-square maps
                # maybe even arbitrary maps or something like that
                g = build_hetgraph(env.halfDG, env.num_tasks,
                                   env.num_robots, env.dur,
                                   map_width, np.array(env.loc, dtype=np.int64),
                                   1.0, # loc_dist_threshold
                                   env.partials, unsch_tasks, rob_chosen,
                                   valid_tasks)
                g = g.to(torch_dev)
                featd = hetgraph_node_helper(env.halfDG.number_of_nodes(),
                                             env.partialw, env.partials,
                                             env.loc, env.dur, map_width,
                                             env.num_robots,
                                             len(valid_tasks))
                featd_tensr = {}
                for k in featd:
                    featd_tensr[k] = torch.Tensor(featd[k]).to(torch_dev)
                task_chosen, pre = gnn_pick_task(g, valid_tasks,
                                                 policy_net, featd_tensr)
                if task_chosen >= 0:
                    task_dur = env.dur[task_chosen-1][rob_chosen]
                    rt, reward, done = env.insert_robot(task_chosen,
                                                        rob_chosen)
                    decision_step += 1
                    robots.update_status(task_chosen, rob_chosen,
                                         task_dur, t)
                    print(('Step: %d,Time: %d, Robot %d,'
                           ' Task %02d, Dur %02d')
                            %(decision_step, t, rob_chosen+1,
                              task_chosen, task_dur))
                    print(valid_tasks)
                    print(pre)
                    # uncomment if you want to save intermediate plots
                    # save_plot_act(decision_step, rob_chosen+1,
                    #               valid_tasks, pre, task_chosen, env.loc,
                    #               'plots')
                    # Check for termination
                    if rt == False:
                        print('Infeasible after %d insertions'
                              % (len(env.partialw)-1))
                        terminate = True
                        break
                    elif env.partialw.shape[0]==(env.num_tasks+1):
                        feas_count += 1
                        dqn_opt = env.min_makespan
                        print('Feasible solution found, min makespan: %f' 
                            % (env.min_makespan))
                        terminate = True
                        break

                    # Attempt to pick another robot
                    rob_chosen = robots.pick_robot_by_min_dur(t, env, 'v1',
                                                              exclude)
                else:
                    # No valid tasks for this robot, move to next
                    exclude.append(rob_chosen)
                    rob_chosen = robots.pick_robot_by_min_dur(t, env, 'v1',
                                                              exclude)
            else:
                break
        if terminate:
            break
    # construct global plan
    # list of [start, stop, robot, location]
    env.loc  # this stores locations
    schedule = []
    for rob_id in range(env.num_robots):
        rob = robots.robots[rob_id].id
        print('Robot %d' % rob)
        for task in robots.robots[rob_id].schedule:
            print('Task (%d,%d,%d)'%(task.id, task.start_time, task.end_time))
            schedule.append([task.start_time, task.end_time, rob, task.id])
    return schedule
    

fname = './data/00374'

# initialize the scheduling enviroment with data files


# size of location map, 3x3 is used
map_width = 3

"""TODO:

Add support for (x, y) locations
In order to achieve this, some code in SchedulingEnv will have to be changed
Maybe this will impact more things
Now everything is restricted to square grids of tasks
"""


print('Ready')

print('Loading model')
model = load_model('./checkpoint.tar')

plan = schedule(fname, model, map_width)
