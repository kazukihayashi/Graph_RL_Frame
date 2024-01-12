import numpy as np
np.random.seed(0)
import copy
import csv
import time
import FrameEnv

### User specified parameters ###
N_STEP = 1000
#################################

env = FrameEnv.Frame()

'''
Optimization class using the trained machine learning model
'''
class GERL(): # The name represents Graph Embedding + Reinforcement Learning
    def __init__(self):
        import Agent
        v,w,C,infeasible_action = env.reset(test=True)
        self.agent = Agent.Agent(v.shape[1],w.shape[1],100,infeasible_action.shape[1],use_gpu=False)
        self.agent.brain.model.Load(filename="trained_model_{0}".format(FrameEnv.__name__))
        while True:
            action, _ = self.agent.get_action(v,w,C,0.0,infeasible_action)
            v, w, _ ,ep_end, infeasible_action = env.step(action)
            if ep_end:
                break
        sec_num = np.copy(env.sec_num)
        if action[1] == 0:
            sec_num[action[0]] += 50
        A = np.array([env.column_section_list[sec_num[i]] for i in range(env.n_column)]+[env.beam_section_list[sec_num[i]] for i in range(env.n_column,env.nm)],dtype=float)[:,0]
        V = np.sum(env.length*A)
        print("Volume: {:}".format(V))
        return

'''
Optimization with the trained machine learning model
'''
# nGERL = 10
# t1 = time.time()#time.process_time()
# for i in range(nGERL):
#     GERL()
# t2 = time.time()#time.process_time()
# print("Elapsed time: {:}".format((t2-t1)/nGERL))

'''
Optimization class using conventional particle swarm optimization (PSO)
'''
class PSO():
    def __init__(self,init_X,init_F,bounds,np=10):
        self.init_X = init_X
        self.init_F = init_F
        self.nvar = bounds.shape[0]
        self.bounds = bounds
        self.np = np
        self.Reset()
        return

    def Reset(self):
        self.p_best_f = None
        self.f_best_f = np.Inf
        self.swarm = []
        self.history = []
        for _ in range(self.np):
            self.swarm.append(Particle(self.init_X,self.init_F,self.bounds))
        self.p_best_g = self.init_X
        self.f_best_g = self.init_F
        self.c_best_g = False # constriant function satisfaction
        return

    def Optimize(self,Func,seed,n_iter=2000):

        np.random.seed(seed)
        self.Reset()
        iteration = 0
        while iteration < n_iter:
            for i in range(len(self.swarm)):
                self.swarm[i].p_i[self.swarm[i].p_i<0] = self.bounds[self.swarm[i].p_i<0,0]
                self.swarm[i].p_i[self.swarm[i].p_i>1] = self.bounds[self.swarm[i].p_i>1,1]
                self.swarm[i].Update(self.p_best_g,iteration/n_iter)
                
            for i in range(len(self.swarm)): 
                F, feasible = Func(self.swarm[i].p_i)
                iteration += 1
                if F < self.swarm[i].f_best_i:
                    self.swarm[i].p_best_i = copy.copy(self.swarm[i].p_i)
                    self.swarm[i].f_best_i = F
                if F < self.f_best_g:
                    self.p_best_g = copy.copy(self.swarm[i].p_i)
                    self.f_best_g = F
                    self.c_best_g = feasible
            self.history.append(self.f_best_g)
            # print(self.f_best_g, self.c_best_g)
                            
        return self.p_best_g, self.f_best_g, self.c_best_g, self.history

class Particle():
    def __init__(self,init_X,init_F,bounds):
        self.nvar = np.size(bounds,0)
        self.bounds = bounds
        self.range = bounds[:,1] - bounds[:,0]

        self.p_i = init_X
        self.v_i = (np.random.rand(self.nvar)-0.5)*self.range*0.1
        self.p_best_i = init_X
        self.f_best_i = init_F

        self.C1 = 2.0
        self.C2 = 2.0
        return

    def Update(self, p_best_g, progress):
        v_cognitive = self.C1 * np.random.rand(self.nvar) * (self.p_best_i-self.p_i)
        v_social = self.C2 * np.random.rand(self.nvar) * (p_best_g-self.p_i)
        v = 0.7 * self.v_i + v_cognitive + v_social

        p_i_before = copy.copy(self.p_i)
        self.p_i = self.p_i + v

        l = self.p_i < self.bounds[:,0]
        self.p_i[l] = self.bounds[l,0]
        u = self.p_i > self.bounds[:,1]
        self.p_i[u] = self.bounds[u,1]

        self.v_i = self.p_i - p_i_before
        return

'''
Optimization with PSO
'''
nPSO = 10
# all_history = []
min_history = []

t1 = time.time() #time.process_time()

env.reset(test=True)
init_X = np.ones(env.nm)*0.5
init_F,_ = env.func(init_X)
bounds = np.zeros((env.nm,2))
bounds[:,1] = 1.0

for i in range(nPSO):
    optimizer = PSO(init_X,init_F,bounds)
    p,f,c,h = optimizer.Optimize(env.func,i*1000)
    # all_history.append(optimizer.history)
    if len(min_history) == 0 or f < np.min(min_history):
        env.func(p)
        env.render()
    min_history.append(np.min(optimizer.history))

t2 = time.time() #time.process_time()
print("process time:{:} seconds".format((t2-t1)/nPSO))

# with open('opt_history.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerows(all_history)

print("Maximum objfun is",np.max(min_history))
print("Median of objfun is",np.median(min_history))
print("Minimum objfun is",np.min(min_history))
print("Average objfun is",np.mean(min_history))
print("Std.dev. objfun is",np.std(min_history))