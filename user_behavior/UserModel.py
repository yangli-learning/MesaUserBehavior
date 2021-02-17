from mesa import Agent, Model
from mesa.time import RandomActivation
import matplotlib.pyplot as plt
from mesa.space import MultiGrid
import numpy as np
from mesa.datacollection import DataCollector
from mesa.batchrunner import BatchRunner
from sortedcollections import ValueSortedDict
from pdb import set_trace as bp
from sklearn.preprocessing import normalize
import collections
from functools import partial


# parameter: p_participate
class UserAgent(Agent):
    """A user agent
    Parameters:
        unique_id: ID
        model: model that stores the agent
        query_size: sliding window size
        p_part: probability an agent will participate in a task (active=True)
    """
    def __init__(self, unique_id, model,queue_size=None,p_part=None):
        super().__init__(unique_id, model)
        self.queue_size =  queue_size if queue_size else model.queue_size
        self.p_participate = p_part if p_part else model.p_part

        # a buffer to record the participation pos in the recent past
        self.traj_queue =  collections.deque([], maxlen=self.queue_size)
        # whether the agent is active
        self.is_active = False

        print('add agent',self.unique_id,'queue_size=',self.queue_size,
            'p_part=',self.p_participate)

    def step(self):
        """ for a probability p, user will select an activity to participate"""
        if self.random.uniform(0,1) < self.p_participate:
            self.is_active  = True
            self.join()
        else:
            self.is_active = False
            self.leave()

    def count_in_dq(self,dq, item):
        """count the frequency of item in a deque"""
        return sum(elem == item for elem in dq)

    def join(self):
        """
        move to the first available spot from visited locations in the past time
        window, ordered by visiting frequency; if no space is available,
        move to a random empty location
        """
        candidates = ValueSortedDict({pos: self.count_in_dq(self.traj_queue,pos) \
            for pos in self.traj_queue} )

        success = False
        while candidates:
            pos,_ = candidates.popitem()
            # move to this location, if success
            success = self.move(col=pos[0],row=pos[1])
            if success:
                self.traj_queue.appendleft(pos)

                break
        candidates = [ (i,j) for i in range(self.model.grid.width) for j in range(self.model.grid.height) ]

        while not success:
            # move to a random open location
            pos = self.random.choice(candidates)
            #print('pos',pos)
            success = self.move( col = pos[0], row=pos[1] )

            if success:
                self.traj_queue.appendleft(pos)

    def leave(self):
        """remove record from current pos"""
        if self.pos:
            self.model.grid.remove_agent(self)

    def move(self,row,col):
        """attempt to move to the specific location,
        Parameters:
            row,col: index of target position
        Return:
            False if no space is available,, otherwise return True
        """
        agents = self.model.grid.get_cell_list_contents([ (col,row)])
        success = False
        if len(agents)< self.model.grid.max_agent_per_cell:
            if not self.pos :
                self.model.grid.place_agent(self,(col,row))
            else:
                self.model.grid.move_agent(self, (  col,row ))
            success =  True
        return success


class UserModel(Model):
    """A model with some number of agents.
    Parameters:
        N: # of agents
        width, height:  width and height of grid
        max_agent_per_cell: max number of agents in each grid cell
                        (# of people necessary for each task)
        query_size: sliding window size
        p_part: probability an agent will participate in a task (active=True)
    """
    def __init__(self, N,width,height,max_agent_per_cell=3,
                 queue_size=12,p_part=1):
        self.num_agents = N
        self.grid = MultiGrid(width=width, height=height, torus=False)
        self.grid.max_agent_per_cell = max_agent_per_cell
        self.queue_size=queue_size
        self.p_part = p_part
        self.schedule = RandomActivation(self)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            a = UserAgent(i, self )
            self.schedule.add(a)

        # Create data collector
        self.datacollector = DataCollector(\
            model_reporters = {"H(pos|user)": [ compute_cond_entropy,[self, 'None']],
            "H(pos|user)/H(pos)":[ compute_cond_entropy,[self, 'asymmetric']],
            "H(pos|user)/(H(pos)+H(user))": [compute_cond_entropy,[self, 'symmetric']] },
            agent_reporters= {"agent pos":"pos"})


    def step(self):
        """ one model step  """
        self.datacollector.collect(self)
        self.schedule.step()

def count_pos_from_traj(model,traj_queue):
    """ count position frequency from past agent trajectory
    Parameter:
        traj_queue: an iterable object containing agent positions
    Return:
        vectorized visit frequency at each grid location
    """
    freq = np.zeros((model.grid.width, model.grid.height))
    for pos in traj_queue:
        freq[pos[0]][pos[1]]+=1
    return freq.flatten()

def compute_cond_entropy(model,normalizer= "asymmetric"):
    """
    compute the conditional entropy of position (pos) given user H(pos|user)
     in the most recent time window of length model.queue_size.

    Parameter:
        model: the simulation model
        normalizer: "None" is unnormalized entropy, "asymmetric" is normalized
            using H(pos), symmetric is normalized using H(pos)+H(user)
    """
    H_pos_given_user = np.zeros(model.num_agents)
    p_user = np.zeros(model.num_agents)
    p_pos = np.zeros(model.grid.width* model.grid.height)
    for i,agent in enumerate(model.schedule.agents):
        agent_history =  count_pos_from_traj(model,agent.traj_queue)
        p_user[i] = np.sum(agent_history)
        p_pos += agent_history
        p_pos_given_user = normalize(agent_history[:,np.newaxis],
                                     axis=0,norm='l1').ravel() + 1e-9
        H_pos_given_user[i]  = -1* p_pos_given_user.dot(np.log(p_pos_given_user))
    p_user = normalize(p_user[:,np.newaxis],axis=0,norm='l1').ravel()+ 1e-9
    p_pos = normalize(p_pos[:,np.newaxis],axis=0,norm='l1').ravel()+ 1e-9
    H_pos_given_users= p_user.dot(H_pos_given_user)
    H_pos = -1* p_pos.dot(np.log(p_pos) )
    H_user = -1 *p_user.dot(np.log(p_user))
    if normalizer=='asymmetric':
        NCE = H_pos_given_users/H_pos
    elif normalizer=='symmetric':
        NCE = H_pos_given_users/(H_pos+H_user)
    else:
        NCE = H_pos_given_users
    return NCE



def singleRun():
    task_size = 3 # max user per task
    N=100 # number of users
    p_part=1
    queue_size = 12
    model = UserModel(N,10,10,
                    max_agent_per_cell = task_size,
                    queue_size=queue_size,
                    p_part=p_part
                    )
    for iter in range(1000):
        if iter % 50==0:
            print(iter)
        model.step()

    # get final position of agents
    agent_counts = np.zeros((model.grid.width, model.grid.height))
    for cell in model.grid.coord_iter():
        cell_content, x, y = cell
        agent_count = len(cell_content)
        agent_counts[x][y] = agent_count
    print('max agent count=',np.max(agent_counts))
    plt.imshow(agent_counts, interpolation='nearest')
    plt.colorbar()
    plt.show()


    # plot the conditional entropy score
    cond_entropy = model.datacollector.get_model_vars_dataframe()
    cond_entropy.plot()
    plt.savefig('output/entropy_N%d_ts%d_w%d_p%f.pdf' % (N,task_size,
                                                        queue_size,p_part))
    plt.show()

    # save agent position to file
    dataframe= model.datacollector.get_agent_vars_dataframe()
    print(dataframe)
    dataframe.to_csv('output/agent_pos_N%d_ts%d_w%d_p%f.csv' % (N,task_size,
                                                        queue_size,p_part))


if __name__ == '__main__':
    singleRun()
