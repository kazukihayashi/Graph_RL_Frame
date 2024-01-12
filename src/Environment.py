import numpy as np
np.random.seed(1000)
from copy import deepcopy
import csv
import Agent
import Plotter
from os import path, makedirs

### User specified parameters ###
import FrameEnv as env
N_FEATURE = 100
RECORD_INTERVAL = 10
MAX_STEPS = 10000
#################################

class Environment():
    def __init__(self,gpu,mode):
        self.mode = mode
        self.env = env.Frame(self.mode)
        v,w,_,infeasible_action = self.env.reset(test=True)
        self.agent = Agent.Agent(v.shape[1],w.shape[1],N_FEATURE,infeasible_action.shape[1],gpu)
        self.agent.brain.model.Load(filename="trained_model_{0}_{1}".format(env.__name__,self.mode))
        if gpu:
            self.agent.brain.model = self.agent.brain.model.to("cuda")
        pass

    def Train(self,n_episode):

        history = np.zeros(n_episode//RECORD_INTERVAL,dtype=float)
        top_score = -np.inf
        top_scored_iteration = -1
        top_scored_model = deepcopy(self.agent.brain.model)

        for episode in range(n_episode):

            v,w,C,infeasible_action = self.env.reset()
            total_reward = 0.0
            aveQ = 0.0
            aveloss = 0.0
            for t in range(MAX_STEPS):
                action,q = self.agent.get_action(v,w,C,0.1,infeasible_action)
                aveQ += q
                v_next, w_next, reward, ep_end, infeasible_action = self.env.step(action)
                self.agent.memorize(C,v,w,action,reward,v_next,w_next,ep_end,infeasible_action)
                v = np.copy(v_next)
                w = np.copy(w_next)
                aveloss += self.agent.update_q_function()
                total_reward += reward
                if ep_end:
                    break

            print("episode {0:<4}: step={1:<3} reward={2:<+5.1f} aveQ={3:<+7.2f} loss={4:<7.2f}".format(episode,t+1,total_reward,aveQ/(t+1),aveloss/(t+1)))
            if episode % RECORD_INTERVAL == RECORD_INTERVAL-1:
                v, w, C, infeasible_action = self.env.reset(test=True)
                total_reward = 0.0
                for t in range(MAX_STEPS):
                    action, _ = self.agent.get_action(v,w,C,0.0,infeasible_action)
                    v, w, reward, ep_end, infeasible_action = self.env.step(action)
                    total_reward += reward
                    if ep_end:
                        break
                if(total_reward >= top_score):
                    top_score = total_reward
                    top_scored_iteration = episode
                    top_scored_model = deepcopy(self.agent.brain.model)
                    
                history[episode//RECORD_INTERVAL] = total_reward

        
        if not path.exists('result'):
            makedirs('result')

        with open("result/info.txt", 'w') as f:
            f.write(str.format("top-scored iteration: {0} \n",top_scored_iteration+1))

        top_scored_model.Save(filename="trained_model_{0}_{1}".format(env.__name__,self.mode))

        Plotter.graph(history)

        with open("result/reward.csv", 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(history)       

    def Test(self,test_model):
        
        v, w, C, infeasible_action = self.env.reset(test=test_model)
        self.agent = Agent.Agent(v.shape[1],w.shape[1],N_FEATURE,infeasible_action.shape[1],False)
        self.agent.brain.model.Load(filename="trained_model_{0}_{1}".format(env.__name__,self.mode))
        self.env.render()
        total_reward = 0.0

        for i in range(MAX_STEPS):
            print('step:'+str(i+1))
            action, _ = self.agent.get_action(v,w,C,0.0,infeasible_action)
            v, w, reward ,ep_end, infeasible_action = self.env.step(action)
            # print(action)
            # if i > 384:
                # self.env.render()
            total_reward += reward
            if ep_end:
                self.env.render()
                break
