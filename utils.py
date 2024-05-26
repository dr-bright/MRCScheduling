import dgl, json, yaml, sys, torch, torch.nn as nn, numpy as np
import sys, networkx as nx, numpy.typing as npt

from collections import Counter
from typing import Tuple
from benchmark.JohnsonUltra import johnsonU
from benchmark.edfutils import RobotTeam
from graph.hetgat import HeteroGATLayer

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
    pass


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
    pass


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


class Scheduler:
    # TODO: implement step inspection for viz purposes
    # TODO: remote edfutils dependency and make this class self-contained
    # TODO: bring in create_model function
    """A wrapper around GNN trained for MARS logistic routing
    
    Automates model construction and model loading,
              map and tasks description parsing,
              plan building and export
    """
    C = 3.0    # discount factor for reward calculation
    dur: npt.NDArray[np.float64]
    ddl: list[tuple[int, float]]
    wait: list[tuple[int, int, float]]
    pos: list[tuple[float, float]]
    loc: list[int]
    near_threshold: float

    @staticmethod
    def create_model(checkpoint_tar=None, device='cpu'):
        # device = torch.device('cuda')
        # use this for CPU-only mode
        device = torch.device(device)

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
        if checkpoint_tar is not None:
            cp = torch.load(checkpoint_tar, map_location=device)
            model.load_state_dict(cp['policy_net_state_dict'])
        return model

    def __init__(self, model=None, device='cpu'):
        #TODO if model is a checkpoint path, load it
        if isinstance(model, str):
            model = self.create_model(model, device)
        self.model = model
        self.device = device
    
    def fetch_tasks_legacy(self, tasks_path, near_threshold = 1.0):
        # load constraints
        self.dur = np.loadtxt(tasks_path+'_dur.txt', dtype=np.float64)
        self.ddl = np.loadtxt(tasks_path+'_ddl.txt', dtype=np.float64).reshape(-1, 2).tolist()
        self.ddl = [(round(task_id), cstr) for task_id, cstr in self.ddl]
        self.wait = np.loadtxt(tasks_path+'_wait.txt', dtype=np.float64).reshape(-1, 3).tolist()
        self.wait = [(round(dep_id), round(task_id), cstr) for dep_id, task_id, cstr in self.wait]
        # reshape if shape is 1D, meaning there is only one constraint
        # if len(self.ddl) > 0 and len(self.ddl.shape) == 1:
        #     self.ddl = self.ddl.reshape(1, -1)
        # if len(self.wait) > 0 and len(self.wait.shape) == 1:
        #     self.wait = self.wait.reshape(1, -1)
        loc = np.loadtxt(tasks_path+'_loc.txt', dtype=np.float64).tolist()
        loc = list(map(tuple, loc))
        self.pos = list(set(loc))
        self.loc = list(map(self.pos.index, loc))
        self.near_threshold = near_threshold
    
    def fetch_tasks(self, tasks: dict | str): # type: ignore
        if isinstance(tasks, str):
            if tasks.lower().endswith('.json'):
                with open(tasks, 'rt', encoding='utf-8') as f:
                    tasks = json.load(f)
            else:
                with open(tasks, 'rt', encoding='utf-8') as f:
                    tasks = yaml.safe_load(f)
            tasks: dict
        num_robots = int(tasks.get('robots', 0))
        self.near_threshold = float(tasks.get('near_threshold', 1.0))
        if num_robots == 0:
            for task in tasks['tasks']:
                if np.iterable(task['dur']):
                    num_robots = len(task['dur'])
                    break
            if num_robots == 0:
                raise ValueError('Task description is malformed:'
                                ' cant infer num_robots')
        self.dur = np.zeros((len(tasks['tasks']), num_robots))
        self.ddl = []
        self.wait = []
        self.pos = list()
        self.loc = list()
        for task_id, task in enumerate(tasks['tasks']):
            self.dur[task_id] = task['dur']
            if 'ddl' in task:
                self.ddl.append((task_id, float(task['ddl'])))
            wait = set(task.keys()) - {'pos', 'dur', 'ddl'}
            for dep in wait:
                dep_id = int(dep)
                self.wait.append((dep_id, task_id, float(task[dep])))
            pos = task['pos']
            try:
                pos_id = self.pos.index(pos)
            except ValueError:
                pos_id = len(self.pos)
                self.pos.append(pos)
            self.loc.append(pos_id)

    def dump_tasks(self, saveto=None):
        tasks = {
            'robots': len(self.dur[0]),
            'near_threshold': self.near_threshold,
            'tasks': []
        }
        for task_id, dur in enumerate(self.dur):
            dur: npt.NDArray[np.float64]
            tasks['tasks'].append({
                'pos': [*self.pos[self.loc[task_id]]],
                'dur': dur.tolist()
                })
        for ddl in self.ddl:
            task_id, cstr = ddl
            tasks['tasks'][task_id]['ddl'] = cstr
        for wait in self.wait:
            dep_id, task_id, cstr = wait
            tasks['tasks'][task_id][dep_id] = cstr
        if saveto is None:
            return tasks
        elif isinstance(saveto, str):
            with open(saveto, 'wt', encoding='utf-8') as f:
                if saveto.lower().endswith('.json'):
                    json.dump(tasks, f, indent=4)
                else:
                    yaml.safe_dump(tasks, f, allow_unicode=True, )
        else:
            json.dump(tasks, saveto)

    @property
    def num_tasks(self):
        return len(self.dur)

    @property
    def num_robots(self):
        return len(self.dur[0])

    @property
    def max_deadline(self):
        # TODO: think of a better way of computing this
        # there must be something
        # this is a shitty euristic
        # return self.num_tasks * 10
        # what about summing max_dur + all waits ?
        val = self.dur.max(axis=1).sum()
        for _, _, cstr in self.wait:
            val += cstr
        return val

    @property
    def M(self):
        """Infeasible reward token"""
        return self.num_tasks * 10.0
    
    def initialize(self):
        """Performs state and STN initialization"""
        # initial partial solution with t0
        # t0 appears in all partial schedules
        self.partials = []
        for i in range(self.num_robots):
            self.partials.append(np.zeros(1, dtype=np.int32))
        
        self.partialw = np.zeros(1, dtype=np.int32)
        
        # maintain a graph with min/max duration for unscheduled tasks
        # Initialize directed graph    
        self.g = DG = nx.DiGraph()
        DG.add_nodes_from(['s000', 'f000'])
        DG.add_edge('s000', 'f000', weight = self.max_deadline)
        # Add task nodes
        for i in range(1, self.num_tasks+1):
            # Add si and fi
            si = 's%03d' % i
            fi = 'f%03d' % i
            DG.add_nodes_from([si, fi])
            DG.add_weighted_edges_from([(si, 's000', 0),
                                        ('f000', fi, 0)])
        # Add task durations
        for task_id, dur in enumerate(self.dur):
            si = 's%03d' % (task_id+1)
            fi = 'f%03d' % (task_id+1)
            dur_min = min(dur)
            dur_max = max(dur)
            DG.add_weighted_edges_from([(si, fi, dur_max),
                                        (fi, si, -1 * dur_min)])
        # Add deadlines
        for ddl in self.ddl:
            ti, ddl_cstr = ddl
            fi = 'f%03d' % ti
            DG.add_edge('s000', fi, weight = ddl_cstr)            
        # Add wait constraints
        for wait in self.wait:
            ti, tj, wait_cstr = wait
            si = 's%03d' % ti
            fj = 'f%03d' % tj
            DG.add_edge(si, fj, weight = -1 * wait_cstr)
        
        # get initial min make span
        success, min_makespan = self.check_consistency_makespan()
        if success:
            self.min_makespan = min_makespan
        else:
            print('Initial STN infeasible.')
    
    def check_consistency_makespan(self, updateDG = True):
        '''Check consistency and get min make span
        Also creates the half min graph
        '''
        consistent = True
        try:
            p_ultra, d_ultra = johnsonU(self.g)
        except Exception as e:
            consistent = False
            print('Infeasible:', e)
        # Makespan
        # Only consider the last finish time of scheduled tasks
        if consistent:        
            if len(self.partialw) == 1:
                min_makespan = 0.0
            else:
                tmp = []
                for i in range(1,len(self.partialw)):
                    ti = self.partialw[i]
                    fi = 'f%03d' % ti
                    tmp.append(-1.0 * d_ultra[fi]['s000'])
    
                tmp_np = np.array(tmp)
                min_makespan = tmp_np.max()
        else:
            min_makespan = self.M
            return consistent, min_makespan
        
        if not updateDG:
            return consistent, min_makespan
        # Min distance graph & Half min graph
        juDG = nx.DiGraph()
        for i in range(0, self.num_tasks+1):
            # Add si and fi
            si = 's%03d' % i
            fi = 'f%03d' % i
            # minDG.add_nodes_from([si, fi])
            if i == 0:
                juDG.add_nodes_from([si, fi])
            else:
                juDG.add_node(si)
        # add shortest path distance edges
        for k_start in d_ultra:
            for k_end in d_ultra[k_start]:
                # print(key_start, key_end)
                # check if path is inf
                if d_ultra[k_start][k_end] < 9999:
                    # minDG.add_edge(k_start, k_end, 
                    #                weight = d_ultra[k_start][k_end])
                    if juDG.has_node(k_start) and juDG.has_node(k_end):
                        juDG.add_edge(k_start, k_end,
                                      weight = d_ultra[k_start][k_end])
        # self.minDG = minDG
        self.halfDG = juDG
        return consistent, min_makespan
    
    def insert_robot(self, ti, rj, updateDG = True):
        '''...

        ti is task number 1~num_tasks
        rj is robot number 0~num_robots-1
        append ti to rj's partial schedule
        also update the STN
        '''
        # sanity check
        if rj < 0 or rj >= self.num_robots:
            raise RuntimeError('invalid insertion')  
        # find tj and update partial solution
        # tj is the last task of rj's partial schedule
        # insert ti right after tj
        tj = self.partials[rj][-1]
        self.partials[rj] = np.append(self.partials[rj], ti)
        self.partialw = np.append(self.partialw, ti)
        # update graph
        # insert ti after tj, no need to add when tj==0    
        # no need to insert if a wait constraint already exists
        if tj != 0:
            si = 's%03d' % ti
            fj = 'f%03d' % tj
            if not self.g.has_edge(si, fj):
                self.g.add_edge(si, fj, weight = 0)
        # [New] Also, replace the task duration of ti with actual duration
        si = 's%03d' % ti
        fi = 'f%03d' % ti
        ti_dur = self.dur[ti-1][rj]
        # this will rewrite previous edge weights
        self.g.add_weighted_edges_from([(si, fi, ti_dur),
                                        (fi, si, -1 * ti_dur)])
        # make sure start time of all unscheduled tasks is >= t si
        for k in range(1, self.num_tasks+1):
            if k not in self.partialw:
                # tk starts no earlier than si
                # si <= sk, si-sk<=0, sk->si:0
                si = 's%03d' % ti
                sk = 's%03d' % k
                if not self.g.has_edge(sk, si):
                    self.g.add_edge(sk, si, weight = 0)
        # make sure the start time of all unscheduled tasks that
        # are within the allowed distance (diff) happen after fi
        for k in range(1, self.num_tasks+1):
            if k not in self.partialw:
                xi, yi = self.pos[self.loc[ti-1]]
                xk, yk = self.pos[self.loc[k-1]]
                dist_2 = (xi - xk) * (xi - xk) + (yi - yk) * (yi - yk)               
                
                if dist_2 <= self.near_threshold ** 2:
                    # tk starts after fi
                    # fi <= sk, fi-sk <=0, sk->fi:0
                    fi = 'f%03d' % ti
                    sk = 's%03d' % k
                    if not self.g.has_edge(sk, fi):
                        self.g.add_edge(sk, fi, weight=0)
        # calculate reward for this insertion
        # TODO: consider embedding this function
        success, reward = self.calc_reward_discount(updateDG)
        # check done/termination
        if success==False:
            done = True
        elif (self.partialw.shape[0]==self.num_tasks+1):
            done = True
        else:
            done = False
        return success, reward, done
    
    def calc_reward_discount(self, updateDG = True):
        '''Reward R of a state-action pair is defined as the change
        in objective values after taking the action,
        
        R = −1 × (Zt+1 − Zt).
        
        divide Zt by a factor D > 1 if xt is not a termination state

        Z(infeasible) = M
        '''
        success, min_makespan = self.check_consistency_makespan(updateDG)
        if success:    # feasible
            # if last step
            if self.partialw.shape[0]==(self.num_tasks+1):
                delta = min_makespan - self.min_makespan/self.C
            else:      # disounted delta
                delta = (min_makespan - self.min_makespan)/self.C
        else:          # infeasible
            delta = self.M - self.min_makespan/self.C
            min_makespan = self.M
        reward = -1.0 * delta
        self.min_makespan = min_makespan
        return success, reward
    
    def get_duration_on_tasks(self, robot, tasks):
        """Returns durations of a robot on a list of tasks.
        Task ids should be 1-indexed, and robot id should be 0-indexed"""
        assert min(tasks) > 0, 'Tasks should be 1-indexed'
        assert 0 <= robot < self.num_robots, 'Robot should be 0-indexed'

        task_ids = [task - 1 for task in tasks]
        return self.dur[task_ids, robot]

    def get_unscheduled_tasks(self):
        '''Return unscheduled tasks given partialw'''
        unsch_tasks = []
        for i in range(1, self.num_tasks+1):
            if i not in self.partialw:
                unsch_tasks.append(i)
        return np.array(unsch_tasks)

    def get_valid_tasks(self, timepoint):
        '''Return unscheduled tasks given partialw
            plus checking if the task can starts at current timepoint
        '''
        valid_tasks = []
        for i in range(1, self.num_tasks+1):
            if i not in self.partialw:
                # check task start time
                # si->s0: A
                # s0 - si <= A
                # si >= -A
                si = 's%03d' % i
                time_si = -1.0 * self.halfDG[si]['s000']['weight']
                # time_si is the earliest time task i can happen
                if time_si <= timepoint:
                    valid_tasks.append(i)
        return np.array(valid_tasks)
    
    def build_hetgraph_featd(self, rob_chosen, t
                             ) -> Tuple[dgl.DGLGraph,dict[str,torch.Tensor]]:
        """Constructs het graph and its feature tensors."""
        # valid_tasks: available tasks filtered from unsch_tasks
        # loc_dist_threshold: threshold for 2 locs to be considered near
        # t: seems like current time but I don't understand it

        ## BUILD HETGRAPH
        # initialize stuff
        unsch_tasks = np.array(self.get_unscheduled_tasks(), dtype=np.int64)
        valid_tasks = np.array(self.get_valid_tasks(t), dtype=np.int64)
        num_values = len(valid_tasks)
        num_locs = len(self.pos)
        # enumerate 'near' edges
        near_edges = []
        for i, point in enumerate(self.pos):
            for j, neighbour in enumerate(self.pos):
                dist = np.subtract(point, neighbour)
                if dist.dot(dist) <= self.near_threshold ** 2:
                    near_edges.append((i, j))
                    near_edges.append((j, i))
        # initialize hetgraph topology
        num_nodes_dict = {'task':  self.num_tasks + 2,  # TODO: why +2?
                          'loc':   num_locs,
                          'robot': self.num_robots,
                          'state': 1,
                          'value': num_values}
        # sort the nodes and assign an index to each one
        task_name_to_idx = {u: i for i, u
                                 in enumerate(sorted(self.halfDG.nodes))}
        task_edge_to_idx = {(a, b): i for i, (a, b)
                                      in enumerate(self.halfDG.edges)}
        # List of (task id, robot id) tuples
        task_to_robot_data = []
        for rj in range(self.num_robots):
            # add f0
            task_to_robot_data.append((0, rj))
            # add si (including s0)
            for i in range(len(self.partials[rj])):
                ti = self.partials[rj][i].item()
                task_id = ti + 1
                task_to_robot_data.append((task_id, rj))
        # ...
        unsch_task_to_robot = []
        for rj in range(self.num_robots):
            for t in unsch_tasks:
                task_id = t + 1
                unsch_task_to_robot.append((task_id, rj))

        robot_com_data = [(i, j) for i in range(self.num_robots)
                                 for j in range(self.num_robots)]

        data_dict = {
            ('task', 'temporal', 'task'): (
                # Convert named edges to indexes
                [task_name_to_idx[a] for a, _ in self.halfDG.edges],
                [task_name_to_idx[b] for _, b in self.halfDG.edges],
            ),
            ('task', 'located_in', 'loc'): (
                list(range(2, self.num_tasks + 2)),
                self.loc,
            ),
            ('loc', 'near', 'loc'): (
                [i for i, _ in near_edges],
                [j for _, j in near_edges],
            ),
            ('task', 'assigned_to', 'robot'): (
                [task for task, _ in task_to_robot_data],
                [robot for _, robot in task_to_robot_data],
            ),
            ('task', 'take_time', 'robot'): (
                [task for task, _ in unsch_task_to_robot],
                [robot for _, robot in unsch_task_to_robot],
            ),
            ('robot', 'use_time', 'task'): (
                [robot for _, robot in unsch_task_to_robot],
                [task for task, _ in unsch_task_to_robot],
            ),
            ('robot', 'com', 'robot'): (
                [i for i, _ in robot_com_data],
                [j for _, j in robot_com_data],
            ),
            # 4. Add graph summary nodes
            # [task] — [in] — [state]
            ('task', 'tin', 'state'): (
                list(range(self.num_tasks + 2)),
                np.zeros(self.num_tasks + 2, dtype=np.int64),
            ),
            # [loc] — [in] — [state]
            ('loc', 'lin', 'state'): (
                list(range(num_locs)),
                np.zeros(num_locs, dtype=np.int64),
            ),
            # [robot] — [in] — [state]
            ('robot', 'rin', 'state'): (
                list(range(self.num_robots)),
                np.zeros(self.num_robots, dtype=np.int64),
            ),
            # [state] — [in] — [state] self-loop
            ('state', 'sin', 'state'): (
                [0],
                [0],
            ),
            # 5.1 Q value node
            # [task] — [to] — [value]
            ('task', 'tto', 'value'): (
                valid_tasks + 1,
                list(range(num_values)),
            ),
            # [robot] — [to] — [value]
            ('robot', 'rto', 'value'): (
                np.full(num_values, rob_chosen, dtype=np.int64),
                list(range(num_values)),
            ),
            # [state] — [to] — [value]
            ('state', 'sto', 'value'): (
                np.zeros(num_values, dtype=np.int64),
                list(range(num_values)),
            ),
            # [value] — [to] — [value] self-loop
            ('value', 'vto', 'value'): (
                list(range(num_values)),
                list(range(num_values)),
            ),
        }
        graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict,
                                idtype=torch.int64)  # type: ignore
        graph: dgl.DGLGraph
        # Store data of edges by index, as DiGraph.edges.data does not
        # guarantee to have exactly the same ordering as Digraph.edges
        temporal_eweights = torch.zeros((len(self.halfDG.edges), 1),
                                            dtype=torch.float32)
        # Unpack indexes of edge weights
        weights_idx = [task_edge_to_idx[from_node, to_node]
                       for from_node, to_node, _
                       in self.halfDG.edges.data('weight')]  # type: ignore
        # Put weights in tensor according to their indexes
        temporal_eweights[weights_idx, :] = torch.tensor(
            [[w] for _, _, w in self.halfDG.edges.data('weight')],  # type: ignore
            dtype=torch.float32)
        graph.edges['temporal'].data['weight'] = temporal_eweights
        takes_time_weight = torch.zeros(
            (len(unsch_task_to_robot), 1),
            dtype=torch.float32)
        for idx, (task, robot) in enumerate(unsch_task_to_robot):
            # Sub 2 because task 1's node id is 2, but has index 0 in dur
            takes_time_weight[idx] = self.dur[task - 2, robot]  # type: ignore
        graph.edges['take_time'].data['t'] = takes_time_weight
        # Ordering of takes_time and uses_time edges are exactly the same
        graph.edges['use_time'].data['t'] = takes_time_weight.detach().clone()
        graph.to(self.device)
        ## BUILD INITIAL FEATT
        featd = {}
        num_nodes = self.halfDG.number_of_nodes()
        num_locs = len(self.pos)

        # Task features.
        # For scheduled tasks, the feature is [1 0 dur 0 dur 0]
        # For unscheduled ones, the feature is [0 1 min max-min mean std]
        featd['task'] = np.zeros((num_nodes, 6))

        max_dur, min_dur = self.dur.max(axis=1), self.dur.min(axis=1)
        mean_dur, std_dur = self.dur.mean(axis=1), self.dur.std(axis=1)

        # f0
        featd['task'][0, 0] = 1

        # s0~si. s0 has index 1
        for i in range(1, num_nodes):
            ti = i-1
            if ti in self.partialw:
                featd['task'][i, 0] = 1
                if ti > 0:
                    # Ignore s0
                    for j in range(self.num_robots):
                        if ti in self.partials[j]:
                            rj = j
                            break              
                    
                    featd['task'][i, [2, 4]] = self.dur[ti-1][rj]
            else:
                featd['task'][i] = [0, 1, min_dur[ti-1], max_dur[ti-1] - min_dur[ti-1], 
                                        mean_dur[ti-1], std_dur[ti-1]]
        
        # [loc]
        featd['loc'] = np.zeros((num_locs, 1))
        loc_counter = Counter(self.loc)
        for i in range(num_locs):
            # number of tasks in location
            featd['loc'][i, 0] = loc_counter[i]
        
        # [robot]
        featd['robot'] = np.zeros((self.num_robots, 1))
        for i in range(self.num_robots):
            # number of tasks assigned so far
            # including s0
            featd['robot'][i, 0] = len(self.partials[i])
        
        # [state]
        featd['state'] = np.array((num_nodes-1, len(self.partialw),
                                    num_locs, self.num_robots)).reshape(1,4)
        
        # [value]
        featd['value'] = np.zeros((num_values, 1))
        for k in featd:
            featd[k] = torch.Tensor(featd[k]).to(self.device)
        return graph, featd

    def schedule(self, tasks=None, near_threshold=1.0):
        if tasks is not None:
            if isinstance(tasks, str) and not (
                    tasks.lower().endswith('.json')
                    or tasks.lower().endswith('yaml')):
                self.fetch_tasks_legacy(tasks, near_threshold)
            else:
                self.fetch_tasks(tasks)
        robots = RobotTeam(self.num_robots)
        # parameters for logging the solving process
        feas_count = 0
        decision_step = 0
        self.initialize()
        terminate = False
        for t in range(self.num_tasks * 10):
            exclude = []
            rob_chosen = robots.pick_robot_by_min_dur(t, self, 'v1', exclude)
            # Repeatedly select robot with min duration until none available
            while rob_chosen is not None:
                valid_tasks = np.array(self.get_valid_tasks(t),
                                    dtype=np.int64)
                if len(valid_tasks) > 0:
                    graph, featd = self.build_hetgraph_featd(
                        rob_chosen, t)
                    task_chosen, pre = gnn_pick_task(graph, valid_tasks,
                                                    self.model, featd)
                    if task_chosen >= 0:
                        task_dur = self.dur[task_chosen-1][rob_chosen]
                        rt, reward, done = self.insert_robot(task_chosen,
                                                             rob_chosen)
                        decision_step += 1
                        robots.update_status(task_chosen, rob_chosen,
                                            task_dur, t)
                        print(('Step: %d,Time: %d, Robot %d,'
                            ' Task %02d, Dur %02d')
                                %(decision_step, t, rob_chosen+1,
                                task_chosen, task_dur), file=sys.stderr)
                        print(valid_tasks, file=sys.stderr)
                        print(pre, file=sys.stderr)
                        # uncomment to save intermediate plots
                        # save_plot_act(decision_step, rob_chosen+1,
                        #               valid_tasks, pre, task_chosen,
                        #               env.loc, 'plots')
                        # Check for termination
                        if rt == False:
                            print('Infeasible after %d insertions'
                                % (len(self.partialw)-1), file=sys.stderr)
                            terminate = True
                            break
                        elif self.partialw.shape[0]==(self.num_tasks+1):
                            feas_count += 1
                            print(('Feasible solution found,'
                                   ' min makespan: %f')
                                   % self.min_makespan, file=sys.stderr)
                            terminate = True
                            break
                        # Attempt to pick another robot
                        rob_chosen = robots.pick_robot_by_min_dur(
                            t, self, 'v1', exclude)
                    else: # No valid tasks for this robot, move to next
                        exclude.append(rob_chosen)
                        rob_chosen = robots.pick_robot_by_min_dur(
                            t, self, 'v1', exclude)
                else:
                    break
            if terminate:
                break
        # construct global plan
        # list of [start, stop, task_id, robot, position]
        schedule = []
        for rob_id in range(self.num_robots):
            rob = robots.robots[rob_id].id
            print('Robot %d' % rob, file=sys.stderr)
            for task in robots.robots[rob_id].schedule:
                print('Task (%d,%d,%d)'
                      %(task.id, task.start_time, task.end_time), file=sys.stderr)
                task_id = task.id - 1
                schedule.append(
                    [task.start_time, task.end_time,
                     task_id, rob, self.pos[self.loc[task_id]]])
        return schedule

    pass

