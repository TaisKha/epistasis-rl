import numpy as np
import gym
from gym import spaces
import json
from collections import defaultdict
import ptan
import os

import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

EPISODE_LENGTH = 1


class EpistasisEnv(gym.Env):

    def __init__(self):
        self.SAMPLE_SIZE = 600 #t1 = t2 = SAMPLE_SIZE
        self.reset()
        self.action_space = spaces.Box(low=0, high=1, shape=(self.N_SNPS,), dtype=np.uint8)
        self.observation_space = spaces.Box(low=0, high=1, shape=
                        (3, 2*self.SAMPLE_SIZE, self.N_SNPS), dtype=np.uint8)
        
        
    def establish_phen_gen(self, file):
        with open(file) as f:
            data = json.load(f)
            genotype = np.array(data["genotype"])
            self.phenotype = np.array(data["phenotype"])
            self.genotype = genotype.T
            num_phenotypes = max(self.phenotype)+1
            self.disease_snps = data["disease_snps"]
            self.phen_gen = [[] for _ in range(num_phenotypes)]
            for i in range(len(self.genotype)):
                self.phen_gen[self.phenotype[i]].append(i)  
            return  self.genotype.shape[0], self.genotype.shape[1]
        
    def normalize_reward(self, current_reward):
        maximum_env_reward = self._count_reward(self.disease_snps)
        minimal_reward = 0.5
        normalized_reward = (current_reward - minimal_reward) / (maximum_env_reward - minimal_reward)
        if normalized_reward > 1:
            print("normalized reward > 1: \n normalized reward = ", normalized_reward, "\n current reward = ", current_reward, "\n maximum_env_reward = ", maximum_env_reward )
            normalized_reward = 0.1
        return normalized_reward

    
    def step(self, action):
        snp_ids = self._take_action(action)
        reward = self._count_reward(snp_ids)
#         без нормализации
        # reward = self.normalize_reward(reward)
        
        self.current_step += 1
        if self.current_step == EPISODE_LENGTH:
            done = True
        else:
            done = False  
        # done = self.current_step == 1
        obs = None if done else self._next_observation()
        return obs, reward, done, {}
    
    def _count_reward(self, snp_ids):
        
        all_existing_seq = defaultdict(lambda: {'control' : 0, 'case' : 0})
        for i, idv in enumerate(self.obs):
            snp_to_cmp = tuple(idv[snp_id] for snp_id in snp_ids) #tuple of SNP that 
            if self.obs_phenotypes[i] == 0:
                all_existing_seq[snp_to_cmp]['control'] += 1
            else:
                all_existing_seq[snp_to_cmp]['case'] += 1

        #count reward      
        TP = 0 #HR case
        FP = 0 #HR control
        TN = 0 #LR control
        FN = 0 #LR case

        for case_control_count in all_existing_seq.values():
          # if seq is in LR group
            if case_control_count['case'] <= case_control_count['control']: #вопрос <= или <
                FN += case_control_count['case']
                TN += case_control_count['control']
            else:
          # if seq is in HR group
                TP += case_control_count['case']
                FP += case_control_count['control']
        R = (FP + TN) / (TP + FN)
        delta = FP / (TP+0.001)
        gamma = (TP + FP + TN + FN) / (TP+0.001)
        CCR = 0.5 * (TP / (TP + FN) + TN / (FP + TN))
        U = (R - delta)**2 / ((1 + delta) * (gamma - delta - 1 + 0.001))
        koef = 1
        # if len(snp_ids) > len(self.disease_snps):
        #         print("len(snp_ids) > len(self.disease_snps)")
        #         koef = 1 / len(snp_ids)

        return koef*(CCR + U)

  
    def reset(self):
        
        pops = ["ASW", "CEU", "CEU+TSI", "CHD", "GIH", "JPT+CHB", "LWK", "MEX", "MKK", "TSI"]
        sim_idx = np.random.randint(2500)
        corp_idx = np.random.randint(1, 23)
        pop_idx = np.random.choice(pops)
        
        self.filename = f"/home/tskhakharova/epistasis-rl/epigen/sim/{sim_idx}_{corp_idx}_{pop_idx}.json"
        # filename = f"/home/tskhakharova/epistasis-rl/epigen/sim/5_7_CEU.json"
        if not os.path.exists(self.filename):
            os.system(f"cd /home/tskhakharova/epistasis-rl/epigen/ && python3 simulate_data.py --sim-ids {sim_idx} --corpus-id {corp_idx} --pop {pop_idx} --inds 5000 --snps 100 --model models/ext_model.ini")

        self.N_IDV, self.N_SNPS = self.establish_phen_gen(self.filename)
        
        self.obs_phenotypes = None
        one_hot_obs = self._next_observation()
        self.current_step = 0
        
        return one_hot_obs

    def render(self, mode='human', close=False):
        pass
    
    def _take_action(self, action):
        chosen_snp_ids = []
        for i, choice in enumerate(action):
            if choice == 1:
                chosen_snp_ids.append(i)
        return chosen_snp_ids    
    
    def _next_observation(self):
        id_0 = np.random.choice(self.phen_gen[0], self.SAMPLE_SIZE)
        id_1 = np.random.choice(self.phen_gen[1], self.SAMPLE_SIZE)
        sample_ids = np.array(list(zip(id_0,id_1))).flatten()
        self.obs = np.array([self.genotype[idv] for idv in sample_ids])
        self.obs_phenotypes = [self.phenotype[idv] for idv in sample_ids]
        
        #one_hot
        one_hot_obs = F.one_hot(torch.tensor(self.obs), 3)
        one_hot_obs = one_hot_obs.movedim(2, 0)

        return one_hot_obs
    
class FixedEpistasisEnv(gym.Env):

    def __init__(self, sample_size, n_snps, observation_onehot, filename, observation, obs_phenotypes, disease_snps):
        self.one_hot_obs = observation_onehot
        self.filename = filename
        self.obs = observation
        self.obs_phenotypes = obs_phenotypes
        self.disease_snps = disease_snps
        
        self.SAMPLE_SIZE = sample_size #t1 = t2 = SAMPLE_SIZE
        self.N_SNPS = n_snps
        
        self.action_space = spaces.Box(low=0, high=1, shape=(self.N_SNPS,), dtype=np.uint8)
        self.observation_space = spaces.Box(low=0, high=1, shape=
                        (3, 2*self.SAMPLE_SIZE, self.N_SNPS), dtype=np.uint8)
        self.engine = None
        
        
    def normalize_reward(self, current_reward):
        maximum_env_reward = self._count_reward(self.disease_snps)
        minimal_reward = 0.5
        normalized_reward = (current_reward - minimal_reward) / (maximum_env_reward - minimal_reward)
        if normalized_reward > 1:
            print("normalized reward > 1: \n normalized reward = ", normalized_reward, "\n current reward = ", current_reward, "\n maximum_env_reward = ", maximum_env_reward )
            normalized_reward = 0.1
        return normalized_reward

    
    def step(self, action):
        snp_ids = self._take_action(action)
        reward = self._count_reward(snp_ids)
#         без нормализации
        # reward = self.normalize_reward(reward)
        
        self.current_step += 1
        done = self.current_step == EPISODE_LENGTH
        return self.one_hot_obs, reward, done, {}
    
    def _count_reward(self, snp_ids):
        
        if set(snp_ids) == set(self.disease_snps):
            print("Disease snps are found")
            
        
        all_existing_seq = defaultdict(lambda: {'control' : 0, 'case' : 0})
        for i, idv in enumerate(self.obs):
            snp_to_cmp = tuple(idv[snp_id] for snp_id in snp_ids) #tuple of SNP that 
            if self.obs_phenotypes[i] == 0:
                all_existing_seq[snp_to_cmp]['control'] += 1
            else:
                all_existing_seq[snp_to_cmp]['case'] += 1

        #count reward      
        TP = 0 #HR case
        FP = 0 #HR control
        TN = 0 #LR control
        FN = 0 #LR case

        for case_control_count in all_existing_seq.values():
          # if seq is in LR group
            if case_control_count['case'] <= case_control_count['control']: #вопрос <= или <
                FN += case_control_count['case']
                TN += case_control_count['control']
            else:
          # if seq is in HR group
                TP += case_control_count['case']
                FP += case_control_count['control']
        R = (FP + TN) / (TP + FN)
        delta = FP / (TP+0.001)
        gamma = (TP + FP + TN + FN) / (TP+0.001)
        CCR = 0.5 * (TP / (TP + FN) + TN / (FP + TN))
        U = (R - delta)**2 / ((1 + delta) * (gamma - delta - 1 + 0.001))
        koef = 1
        #добавила коэффициент
        if len(snp_ids) > len(self.disease_snps):
                print("len(snp_ids) > len(self.disease_snps)")
                koef = 1 / len(snp_ids)
# отнимаю 0.5
        return koef*(CCR + U - 0.5)

  
    def reset(self):

        self.current_step = 0
        
        return self.one_hot_obs

    def render(self, mode='human', close=False):
        pass
    
    def _take_action(self, action):
        chosen_snp_ids = []
        for i, choice in enumerate(action):
            if choice == 1:
                chosen_snp_ids.append(i)
        return chosen_snp_ids  

class EpiProbabilityActionSelector(ptan.actions.ActionSelector):
    """
    Converts probabilities of actions into action by sampling them
    """
    def __call__(self, probs):
        assert isinstance(probs, np.ndarray)
        assert isinstance(probs[0], np.ndarray)
        actions = []
#         print("EpiProbabilityActionSelector - probs shape:", probs.shape)
        for prob in probs:
            # print(prob, end=' ')
            num_selected_snps = 0
            for oneprob in prob:
                if oneprob > 1/len(prob):
                    num_selected_snps += 1
            wandb.log({"num_selected_snps":num_selected_snps}, commit=False)        
            # print(f'{num_selected_snps=}')        
            if num_selected_snps < 2:
                print("num_selected_snps < 2")
                num_selected_snps = 2
                    
        # for prob in probs:
        #     num_selected_snps = 2
            # num_selected_snps = 0
            # amount_of_oneprob_more_than_1_div_n = 0
            # while amount_of_oneprob_more_than_1_div_n < 2:
            #     amount_of_oneprob_more_than_1_div_n = 0
            #     if num_selected_snps > len(prob)/10:
            #         num_selected_snps = int(len(prob)/10)
            #         break
            #     num_selected_snps += 1
            #     for oneprob in prob:
            #         if oneprob > 1 / num_selected_snps:
            #             amount_of_oneprob_more_than_1_div_n += 1
            
            chosen_snp = np.random.choice(len(prob), size=num_selected_snps, replace=False, p=prob)
            action = np.zeros(len(prob))
            for snp in chosen_snp:
                action[snp] = 1
            actions.append(action)
        return np.array(actions)
    

class SnpPGN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(SnpPGN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float()
#         fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)  
    

GAMMA = 0.99
LEARNING_RATE = 0.01
EPISODES_TO_TRAIN = 32
COUNT = 1000000
WANDB = True
AMOUNT_OF_DATA = 1
SAMPLE_SIZE = 600 #t1 = t2 = SAMPLE_SIZE


def calc_qvals(rewards):
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)
    return list(reversed(res))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temp_env = EpistasisEnv()
    
    fixed_observation_onehot = temp_env.reset()
    fixed_filename = temp_env.filename
    fixed_observation = temp_env.obs
    fixed_obs_phenotypes = temp_env.obs_phenotypes
    fixed_disease_snps = temp_env.disease_snps
    fixed_sample_size = temp_env.SAMPLE_SIZE
    fixed_n_snps = temp_env.N_SNPS
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = FixedEpistasisEnv(fixed_sample_size, fixed_n_snps, fixed_observation_onehot, fixed_filename, fixed_observation, fixed_obs_phenotypes, fixed_disease_snps)

    if WANDB:
        wandb.init(project="epistasis", entity="taisikus", config={
          "learning_rate": LEARNING_RATE,
          "gamma": GAMMA,
          "episodes_to_train": EPISODES_TO_TRAIN,
          "steps_number" : COUNT,
          "data_amount": AMOUNT_OF_DATA,
          "sample_size": SAMPLE_SIZE
        })
        
    net = SnpPGN(env.observation_space.shape, env.N_SNPS)
    net = nn.DataParallel(net)
    net.to(device)
    print(net)
    agent = ptan.agent.PolicyAgent(net, action_selector=EpiProbabilityActionSelector(), apply_softmax=True, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    step_idx = 0
    done_episodes = 0

    batch_episodes = 0
    batch_states, batch_actions, batch_qvals = [], [], []
    cur_rewards = []

    for step_idx, exp in enumerate(exp_source):
        batch_states.append(exp.state)
#         batch_actions.append(int(exp.action))
        batch_actions.append(exp.action)
        cur_rewards.append(exp.reward)

        if exp.last_state is None:
            batch_qvals.extend(calc_qvals(cur_rewards))
            cur_rewards.clear()
            batch_episodes += 1

        # handle new rewards
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            done_episodes += 1
            reward = new_rewards[0]
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))

            if WANDB:
                wandb.log({"reward": reward, "mean_100": mean_rewards, "episodes": done_episodes})
            # print(done_episodes)    
            # if mean_rewards > 0.9:
            #     print("Solved in %d steps and %d episodes!" % (step_idx, done_episodes))
            #     break
            if done_episodes > COUNT:
                print(f"done_episodes > {COUNT}")
                break

        if batch_episodes < EPISODES_TO_TRAIN:
            continue

        optimizer.zero_grad()
        
        states_v = torch.stack(batch_states)
        states_v = states_v.to(device)
        batch_actions_t = torch.FloatTensor(batch_actions)
        batch_actions_t = batch_actions_t.to(device)
        batch_qvals_v = torch.FloatTensor(batch_qvals)
        batch_qvals_v = batch_qvals_v.to(device)

        logits_v = net(states_v)
        log_prob_v = F.log_softmax(logits_v, dim=1)
        
#         print(log_prob_v.shape)
#         print(batch_qvals_v.shape)
#         print(len(batch_states))
#         print(batch_actions_t.shape)
#         log_prob_actions_v = batch_qvals_v * log_prob_v[range(len(batch_states)), batch_actions_t]
        log_prob_actions_v = batch_qvals_v * torch.diagonal(torch.mm(log_prob_v, torch.transpose(batch_actions_t, 0, 1)))
        loss_v = -log_prob_actions_v.mean()

        loss_v.backward()
        optimizer.step()

        batch_episodes = 0
        batch_states.clear()
        batch_actions.clear()
        batch_qvals.clear()
    if WANDB:
        wandb.finish()    

