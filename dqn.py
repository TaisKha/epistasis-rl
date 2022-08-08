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

from types import SimpleNamespace
from typing import Iterable, Tuple, List
import warnings
from datetime import timedelta, datetime

# import ptan_ignite
from ignite.engine import Engine
from ignite.metrics import RunningAverage
# from ignite.contrib.handlers import tensorboard_logger as tb_logger


import enum
import time
from typing import Optional
from ignite.engine import Engine, State
from ignite.engine import Events, EventEnum
from ignite.handlers.timing import Timer
from ignite.contrib.handlers.base_logger import BaseOutputHandler
from ignite.contrib.handlers.wandb_logger import WandBLogger
from typing import Any, Callable, List, Optional, Union

EPISODE_LENGTH = 6000

class OutputHandler(BaseOutputHandler):


    def __init__(
        self,
        tag: str,
        metric_names: Optional[List[str]] = None,
        output_transform: Optional[Callable] = None,
        global_step_transform: Optional[Callable] = None,
        sync: Optional[bool] = None,
        state_attributes: Optional[List[str]] = None,
    ):
        super().__init__(tag, metric_names, output_transform, global_step_transform, state_attributes)
        self.sync = sync

    def __call__(self, engine: Engine, logger: WandBLogger, event_name: Union[str, Events]) -> None:

        if not isinstance(logger, WandBLogger):
            raise RuntimeError(f"Handler '{self.__class__.__name__}' works only with WandBLogger.")

        global_step = self.global_step_transform(engine, event_name)  # type: ignore[misc]
        if not isinstance(global_step, int):
            raise TypeError(
                f"global_step must be int, got {type(global_step)}."
                " Please check the output of global_step_transform."
            )

        metrics = self._setup_output_metrics_state_attrs(engine, log_text=True, key_tuple=False)
        logger.log(metrics, sync=self.sync)


class EpisodeEvents(EventEnum):
    EPISODE_COMPLETED = "episode_completed"
    BOUND_REWARD_REACHED = "bound_reward_reached"
    BEST_REWARD_REACHED = "best_reward_reached"


class EndOfEpisodeHandler:
    def __init__(self, exp_source: ptan.experience.ExperienceSource, alpha: float = 0.98,
                 bound_avg_reward: Optional[float] = None,
                 subsample_end_of_episode: Optional[int] = None):
        """
        Construct end-of-episode event handler
        :param exp_source: experience source to use
        :param alpha: smoothing alpha param
        :param bound_avg_reward: optional boundary for average reward
        :param subsample_end_of_episode: if given, end of episode event will be subsampled by this amount
        """
        self._exp_source = exp_source
        self._alpha = alpha
        self._bound_avg_reward = bound_avg_reward
        self._best_avg_reward = None
        self._subsample_end_of_episode = subsample_end_of_episode

    def attach(self, engine: Engine):
        engine.add_event_handler(Events.ITERATION_COMPLETED, self)
        engine.register_events(*EpisodeEvents)
        State.event_to_attr[EpisodeEvents.EPISODE_COMPLETED] = "episode"
        State.event_to_attr[EpisodeEvents.BOUND_REWARD_REACHED] = "episode"
        State.event_to_attr[EpisodeEvents.BEST_REWARD_REACHED] = "episode"

    def __call__(self, engine: Engine):
        for reward, steps in self._exp_source.pop_rewards_steps():
            engine.state.episode = getattr(engine.state, "episode", 0) + 1
            engine.state.episode_reward = reward
            engine.state.episode_steps = steps
            engine.state.metrics['reward'] = reward
            engine.state.metrics['steps'] = steps
            self._update_smoothed_metrics(engine, reward, steps)
            if self._subsample_end_of_episode is None or engine.state.episode % self._subsample_end_of_episode == 0:
                engine.fire_event(EpisodeEvents.EPISODE_COMPLETED)
            if self._bound_avg_reward is not None and engine.state.metrics['avg_reward'] >= self._bound_avg_reward:
                engine.fire_event(EpisodeEvents.BOUND_REWARD_REACHED)
            if self._best_avg_reward is None:
                self._best_avg_reward = engine.state.metrics['avg_reward']
            elif self._best_avg_reward < engine.state.metrics['avg_reward']:
                engine.fire_event(EpisodeEvents.BEST_REWARD_REACHED)
                self._best_avg_reward = engine.state.metrics['avg_reward']

    def _update_smoothed_metrics(self, engine: Engine, reward: float, steps: int):
        for attr_name, val in zip(('avg_reward', 'avg_steps'), (reward, steps)):
            if attr_name not in engine.state.metrics:
                engine.state.metrics[attr_name] = val
            else:
                engine.state.metrics[attr_name] *= self._alpha
                engine.state.metrics[attr_name] += (1-self._alpha) * val


class EpisodeFPSHandler:
    FPS_METRIC = 'fps'
    AVG_FPS_METRIC = 'avg_fps'
    TIME_PASSED_METRIC = 'time_passed'

    def __init__(self, fps_mul: float = 1.0, fps_smooth_alpha: float = 0.98):
        self._timer = Timer(average=True)
        self._fps_mul = fps_mul
        self._started_ts = time.time()
        self._fps_smooth_alpha = fps_smooth_alpha

    def attach(self, engine: Engine, manual_step: bool = False):
        self._timer.attach(engine, step=None if manual_step else Events.ITERATION_COMPLETED)
        engine.add_event_handler(EpisodeEvents.EPISODE_COMPLETED, self)

    def step(self):
        """
        If manual_step=True on attach(), this method should be used every time we've communicated with environment
        to get proper FPS
        :return:
        """
        self._timer.step()

    def __call__(self, engine: Engine):
        t_val = self._timer.value()
        if engine.state.iteration > 1:
            fps = self._fps_mul / t_val
            avg_fps = engine.state.metrics.get(self.AVG_FPS_METRIC)
            if avg_fps is None:
                avg_fps = fps
            else:
                avg_fps *= self._fps_smooth_alpha
                avg_fps += (1-self._fps_smooth_alpha) * fps
            engine.state.metrics[self.AVG_FPS_METRIC] = avg_fps
            engine.state.metrics[self.FPS_METRIC] = fps
        engine.state.metrics[self.TIME_PASSED_METRIC] = time.time() - self._started_ts
        self._timer.reset()


class PeriodEvents(EventEnum):
    ITERS_10_COMPLETED = "iterations_10_completed"
    ITERS_100_COMPLETED = "iterations_100_completed"
    ITERS_1000_COMPLETED = "iterations_1000_completed"
    ITERS_10000_COMPLETED = "iterations_10000_completed"
    ITERS_100000_COMPLETED = "iterations_100000_completed"


class PeriodicEvents:
    """
    The same as CustomPeriodicEvent from ignite.contrib, but use true amount of iterations,
    which is good for TensorBoard
    """

    INTERVAL_TO_EVENT = {
        10: PeriodEvents.ITERS_10_COMPLETED,
        100: PeriodEvents.ITERS_100_COMPLETED,
        1000: PeriodEvents.ITERS_1000_COMPLETED,
        10000: PeriodEvents.ITERS_10000_COMPLETED,
        100000: PeriodEvents.ITERS_100000_COMPLETED,
    }

    def attach(self, engine: Engine):
        engine.add_event_handler(Events.ITERATION_COMPLETED, self)
        engine.register_events(*PeriodEvents)
        for e in PeriodEvents:
            State.event_to_attr[e] = "iteration"

    def __call__(self, engine: Engine):
        for period, event in self.INTERVAL_TO_EVENT.items():
            if engine.state.iteration % period == 0:
                engine.fire_event(event)



def unpack_batch(batch: List[ptan.experience.ExperienceFirstLast]):
    states, actions, rewards, dones, last_states = [],[],[],[],[]
    for exp in batch:
        state = np.array(exp.state)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            lstate = state  # the result will be masked anyway
        else:
            lstate = np.array(exp.last_state)
        last_states.append(lstate)
    return np.array(states, copy=False), np.array(actions), \
           np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), \
           np.array(last_states, copy=False)

                
def calc_loss_dqn(batch, net, tgt_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = \
        unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)
    state_action_vals = torch.sum(actions_v * net(states_v), dim=1, dtype=torch.float32)
#     actions_v = actions_v.unsqueeze(-1)
    
#     state_action_vals = net(states_v).gather(1, actions_v)
#     state_action_vals = state_action_vals.squeeze(-1)
    with torch.no_grad():
        next_state_vals = tgt_net(next_states_v).max(1)[0]
        next_state_vals[done_mask] = 0.0

    bellman_vals = next_state_vals.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_vals, bellman_vals)

class EpistasisEnv(gym.Env):

    def __init__(self):
        self.SAMPLE_SIZE = 300 #t1 = t2 = SAMPLE_SIZE
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
        # print(f"{snp_ids=}, {self.disease_snps=}")
        reward = self._count_reward(snp_ids)
        # print(f"{reward=}", end=' ')
        reward = self.normalize_reward(reward)
        # print(f"{reward=}")
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
        if len(snp_ids) > len(self.disease_snps):
                print("len(snp_ids) > len(self.disease_snps)")
                koef = 1 / len(snp_ids)

        return koef*(CCR + U)

  
    def reset(self):
        pops = ["ASW", "CEU", "CEU+TSI", "CHD", "GIH", "JPT+CHB", "LWK", "MEX", "MKK", "TSI"]
        sim_idx = np.random.randint(2500)
        corp_idx = np.random.randint(1, 23)
        pop_idx = np.random.choice(pops)
        
        # filename = f"/home/tskhakharova/epistasis-rl/epigen/sim/{sim_idx}_{corp_idx}_{pop_idx}.json"
        filename = f"/home/tskhakharova/epistasis-rl/epigen/sim/5_7_CEU.json"
        if not os.path.exists(filename):
            os.system(f"cd /home/tskhakharova/epistasis-rl/epigen/ && python3 simulate_data.py --sim-ids {sim_idx} --corpus-id {corp_idx} --pop {pop_idx} --inds 5000 --snps 100 --model models/ext_model.ini")

        self.N_IDV, self.N_SNPS = self.establish_phen_gen(filename)
        
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
    
class EpsilonTracker:
    def __init__(self, selector: ptan.actions.EpsilonGreedyActionSelector,
                 params: SimpleNamespace):
        self.selector = selector
        self.params = params
        self.frame(0)

    def frame(self, frame_idx: int):
        eps = self.params.epsilon_start - \
              frame_idx / self.params.epsilon_frames
        self.selector.epsilon = max(self.params.epsilon_final, eps)


def batch_generator(buffer: ptan.experience.ExperienceReplayBuffer,
                    initial: int, batch_size: int):
    buffer.populate(initial)
    while True:
        buffer.populate(1)
        yield buffer.sample(batch_size)


@torch.no_grad()
def calc_values_of_states(states, net, device="cpu"):
    mean_vals = []
    for batch in np.array_split(states, 64):
        states_v = torch.tensor(batch).to(device)
        action_values_v = net(states_v)
        best_action_values_v = action_values_v.max(1)[0]
        mean_vals.append(best_action_values_v.mean().item())
    return np.mean(mean_vals)


def setup_ignite(engine: Engine, params: SimpleNamespace,
                 exp_source, run_name: str, net,
                 extra_metrics: Iterable[str] = ()):
    # get rid of missing metrics warning
    warnings.simplefilter("ignore", category=UserWarning)

    handler = EndOfEpisodeHandler(
        exp_source, bound_avg_reward=params.stop_reward)
    handler.attach(engine)
    EpisodeFPSHandler().attach(engine)

    # @engine.on(EpisodeEvents.EPISODE_COMPLETED)
    # def episode_completed(trainer: Engine):
    #     passed = trainer.state.metrics.get('time_passed', 0)
    #     print("Episode %d: reward=%.5f, steps=%s, "
    #           "speed=%.1f f/s, elapsed=%s" % (
    #         trainer.state.episode, trainer.state.episode_reward,
    #         trainer.state.episode_steps,
    #         trainer.state.metrics.get('avg_fps', 0),
    #         timedelta(seconds=int(passed))))

    @engine.on(EpisodeEvents.BOUND_REWARD_REACHED)
    def game_solved(trainer: Engine):
        passed = trainer.state.metrics['time_passed']
        print("Game solved in %s, after %d episodes "
              "and %d iterations!" % (
            timedelta(seconds=int(passed)),
            trainer.state.episode, trainer.state.iteration))
        trainer.should_terminate = True

    # now = datetime.now().isoformat(timespec='minutes').replace(':', '')
    # wandb.tensorboard.patch(root_logdir="./logs/debug")
    # logdir = f"runs/{now}-{params.run_name}-{run_name}"
    # tb = tb_logger.TensorboardLogger(log_dir=logdir)

    wandb_logger = WandBLogger(
        project="epistasis",
        entity="taisikus",
    )
    
    run_avg = RunningAverage(output_transform=lambda v: v['loss'])
    run_avg.attach(engine, "avg_loss")

    # metrics = ['reward', 'avg_reward']
    metrics = ['reward', 'steps', 'avg_reward']
    event = EpisodeEvents.EPISODE_COMPLETED
    handler = OutputHandler(tag="episodes", metric_names=metrics)
    wandb_logger.attach(engine, log_handler=handler, event_name=event)
    
    # write to tensorboard every 100 iterations
    PeriodicEvents().attach(engine)
    metrics = ['avg_loss', 'avg_fps']
    metrics.extend(extra_metrics)
    handler = OutputHandler(tag="train", metric_names=metrics, output_transform=lambda a: a)
    event = PeriodEvents.ITERS_100_COMPLETED
    wandb_logger.attach(engine, log_handler=handler, event_name=event)
    
    wandb_logger.watch(net)
    
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        
        # print(input_shape)

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
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        # fx = x.float() / 2
        fx = x.float() / 2
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)
    
class EpsilonGreedyActionSelector(ptan.actions.ActionSelector):
    def __init__(self, epsilon=0.05, selector=None):
        self.epsilon = epsilon
        self.selector = selector if selector is not None else ptan.actions.ArgmaxActionSelector()

    def __call__(self, scores):
        assert isinstance(scores, np.ndarray)
        actions = []
        for batch in scores:
            num_selected_snps = 2
            snps_idx = []
            if np.random.random() < self.epsilon:
                snps_idx = np.random.choice(len(batch), num_selected_snps, replace=False)
            else:    
                for i in range(num_selected_snps):
                    largest_score_idx = np.argmax(batch)
                    snps_idx.append(largest_score_idx)
                    batch[largest_score_idx] = -1
            action = np.zeros(len(batch))        
            for snp in snps_idx:
                action[snp] = 1
            actions.append(action)
            
        return np.array(actions)    
        # batch_size, n_actions = scores.shape
        # actions = self.selector(scores)
        # mask = np.random.random(size=batch_size) < self.epsilon
        # rand_actions = np.random.choice(n_actions, sum(mask))
        # actions[mask] = rand_actions
        

params = SimpleNamespace(**{
        'env_name': "EpistasisEnv",
        'stop_reward': 10000,
        'run_name': 'dqn-basic',
        'replay_size': 10 ** 6,
        'replay_initial': 25000,
        'target_net_sync': 10000,
        'epsilon_frames': 500000,
        'epsilon_start': 1.0,
        'epsilon_final': 0.1,
        'learning_rate': 0.00025,
        'gamma': 0.99,
        'batch_size': 32
    })

# # для prelimitary
# params = SimpleNamespace(**{
#         'env_name': "EpistasisEnv",
#         'stop_reward': 1.0,
#         'run_name': 'dqn-basic-5000',
#         'replay_size': 5000,
#         'replay_initial': 100,
#         'target_net_sync': 50,
#         'epsilon_frames': 4500,
#         'epsilon_start': 1.0,
#         'epsilon_final': 0.1,
#         'learning_rate': 0.001,
#         'gamma': 0.99,
#         'batch_size': 32
#     })
NAME = "dqn_baseline"


if __name__ == "__main__":
    
    # wandb.init(project='epistasis', entity="taisikus", sync_tensorboard=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = EpistasisEnv()
    net = DQN(env.observation_space.shape, env.N_SNPS)
    net = nn.DataParallel(net)
    net = net.to(device)
    tgt_net = ptan.agent.TargetNet(net)
    
    selector = EpsilonGreedyActionSelector(
        epsilon=params.epsilon_start)
    
    epsilon_tracker = EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=params.gamma)
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, buffer_size=params.replay_size)
    optimizer = optim.Adam(net.parameters(),
                           lr=params.learning_rate)

    def process_batch(engine, batch):
        optimizer.zero_grad()
        loss_v = calc_loss_dqn(
            batch, net, tgt_net.target_model,
            gamma=params.gamma, device=device)
        loss_v.backward()
        optimizer.step()
        epsilon_tracker.frame(engine.state.iteration)
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()
        return {
            "loss": loss_v.item(),
            "epsilon": selector.epsilon,
        }

    engine = Engine(process_batch)
    setup_ignite(engine, params, exp_source, NAME, net)
    
    engine.run(batch_generator(buffer, params.replay_initial,
                                      params.batch_size))