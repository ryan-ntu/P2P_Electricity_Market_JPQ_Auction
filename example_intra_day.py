from module import micro_grid_agent
from environment import MultiMicroGridEnv
from algorithm.MAPPO import MAPPO

import numpy as np 
import argparse
import os, csv  # 新增：用于保存数据
import torch
import random

# 设置所有随机种子以确保结果可重现
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# 注意：随机种子在解析参数后根据 --seed 或 --seeds 设置

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train', help='train or test')
parser.add_argument('--model_dir', type=str, default='./model', help='模型保存路径')
parser.add_argument('--load_file', type=str, default='./Dataset/load_profiles.npy', help='负载数据')
parser.add_argument('--generation_file', type=str, default='./Dataset/generation_profiles.npy', help='发电数据')

# 电网环境参数
parser.add_argument('--grid1_id', type=int, default=0, help='第1个grid的ID')
parser.add_argument('--grid1_demand', type=float, default=25, help='第1个grid的需求参数')
parser.add_argument('--grid1_generation', type=float, default=5, help='第1个grid的发电参数')
parser.add_argument('--grid1_battery', type=float, default=8, help='第1个grid的电池容量')
parser.add_argument('--grid1_charge', type=float, default=4, help='第1个grid的充电限制')
parser.add_argument('--grid1_discharge', type=float, default=4, help='第1个grid的放电限制')
parser.add_argument('--grid1_storage', type=float, default=0, help='第1个grid的初始储能')

parser.add_argument('--grid2_id', type=int, default=1, help='第2个grid的ID')
parser.add_argument('--grid2_demand', type=float, default=6, help='第2个grid的需求参数')
parser.add_argument('--grid2_generation', type=float, default=7, help='第2个grid的发电参数')
parser.add_argument('--grid2_battery', type=float, default=15, help='第2个grid的电池容量')
parser.add_argument('--grid2_charge', type=float, default=5, help='第2个grid的充电限制')
parser.add_argument('--grid2_discharge', type=float, default=5, help='第2个grid的放电限制')
parser.add_argument('--grid2_storage', type=float, default=2, help='第2个grid的初始储能')

parser.add_argument('--grid3_id', type=int, default=2, help='第3个grid的ID')
parser.add_argument('--grid3_demand', type=float, default=40, help='第3个grid的需求参数')
parser.add_argument('--grid3_generation', type=float, default=10, help='第3个grid的发电参数')
parser.add_argument('--grid3_battery', type=float, default=15, help='第3个grid的电池容量')
parser.add_argument('--grid3_charge', type=float, default=8, help='第3个grid的充电限制')
parser.add_argument('--grid3_discharge', type=float, default=8, help='第3个grid的放电限制')
parser.add_argument('--grid3_storage', type=float, default=0, help='第3个grid的初始储能')

parser.add_argument('--grid4_id', type=int, default=3, help='第4个grid的ID')
parser.add_argument('--grid4_demand', type=float, default=5, help='第4个grid的需求参数')
parser.add_argument('--grid4_generation', type=float, default=15, help='第4个grid的发电参数')
parser.add_argument('--grid4_battery', type=float, default=30, help='第4个grid的电池容量')
parser.add_argument('--grid4_charge', type=float, default=8, help='第4个grid的充电限制')
parser.add_argument('--grid4_discharge', type=float, default=8, help='第4个grid的放电限制')
parser.add_argument('--grid4_storage', type=float, default=20, help='第4个grid的初始储能')


parser.add_argument('--market_common_price', type=float, default=10, help='市场电价')
parser.add_argument('--market_emergency_price', type=float, default=20, help='市场紧急电价')
parser.add_argument('--market_feed_in_price', type=float, default=2, help='市场售电价')
parser.add_argument('--market_max_steps', type=int, default=1440, help='市场最大步数')
parser.add_argument('--market_mechanism', type=str, default='simple', help='市场机制')

# 共有参数
parser.add_argument("--device", type=str, default='cuda:0')
parser.add_argument("--seed", type=int, default=100) # 0 10 100  
parser.add_argument('--seeds', type=int, nargs='*', default=None, help='多随机种子列表，如: --seeds 42 43 44')
parser.add_argument("--max_episodes", type=int, default=int(10000))
parser.add_argument("--save_freq", type=int, default=int(5000//4))
parser.add_argument("--start_steps", type=int, default=0) # 满足此开始更新 此算法不用
parser.add_argument("--random_steps", type=int, default=0)  # 满足此开始自己探索
parser.add_argument("--learn_steps_interval", type=int, default=0) # 这个算法不方便用
# 训练参数
parser.add_argument("--gamma", type=float, default=0.95)
parser.add_argument("--tau", type=float, default=0.01)
## A-C参数   
parser.add_argument("--actor_lr", type=float, default=1e-3)
parser.add_argument("--critic_lr", type=float, default=1e-3)
# PPO独有参数
parser.add_argument("--horizon", type=int, default=720) #
parser.add_argument("--clip_param", type=float, default=0.2)
parser.add_argument("--K_epochs", type=int, default=10) # 15 # 困难任务建议设置为5
parser.add_argument("--entropy_coefficient", type=float, default=0.01)
parser.add_argument("--minibatch_size", type=int, default=512)
parser.add_argument("--lmbda", type=float, default=0.95) # GAE参数
## mappo 参数
parser.add_argument("--huber_delta", type=float, default=10.0) # huber_loss参数
parser.add_argument("--trick", type=dict, default={'adv_norm':False,
                                                        'ObsNorm':False,
                                                        'reward_norm':False,'reward_scaling':False,    # or
                                                        'orthogonal_init':True,'adam_eps':False,'lr_decay':False, # 原代码中设置为False
                                                        # 以上均在PPO_with_tricks.py中实现过
                                                       'ValueClip':False,'huber_loss':False,
                                                       'LayerNorm':True,'feature_norm':False,
                                                       }) 

args = parser.parse_args()

# 设置随机种子：优先使用 --seeds；否则使用 --seed
if args.seeds and len(args.seeds) > 0:
    pass  # 将在各实验循环内逐个设置
else:
    set_seed(args.seed)

action_space = 3

param_panel1 = {
    "id": args.grid1_id,
    "demand_param": args.grid1_demand,
    "generation_param": args.grid1_generation,
    "battery_lim": args.grid1_battery,
    "charge_lim": args.grid1_charge,
    "discharge_lim": args.grid1_discharge,
    "initial_storage": args.grid1_storage,
    "action_space": action_space,
    "load_file": args.load_file,
    "generation_file": args.generation_file
}

param_panel2 = {
    "id": args.grid2_id,
    "demand_param": args.grid2_demand,
    "generation_param": args.grid2_generation,
    "battery_lim": args.grid2_battery,
    "charge_lim": args.grid2_charge,
    "discharge_lim": args.grid2_discharge,
    "initial_storage": args.grid2_storage,
    "action_space": action_space,
    "load_file": args.load_file,
    "generation_file": args.generation_file
}


param_panel3 = {
    "id": args.grid3_id,
    "demand_param": args.grid3_demand,
    "generation_param": args.grid3_generation,
    "battery_lim": args.grid3_battery,
    "charge_lim": args.grid3_charge,
    "discharge_lim": args.grid3_discharge,
    "initial_storage": args.grid3_storage,
    "action_space": action_space,
    "load_file": args.load_file,
    "generation_file": args.generation_file
}

param_panel4 = {
    "id": args.grid4_id,
    "demand_param": args.grid4_demand,
    "generation_param": args.grid4_generation,
    "battery_lim": args.grid4_battery,
    "charge_lim": args.grid4_charge,
    "discharge_lim": args.grid4_discharge,
    "initial_storage": args.grid4_storage,
    "action_space": action_space,
    "load_file": args.load_file,
    "generation_file": args.generation_file
}

grid_1 = micro_grid_agent(param_panel1)

grid_2 = micro_grid_agent(param_panel2)

grid_3 = micro_grid_agent(param_panel3)

grid_4 = micro_grid_agent(param_panel4)
  


env_config = {
    "max_steps": args.market_max_steps,
    "common_price": args.market_common_price,
    "emergency_price": args.market_emergency_price,
    "feed_in_price": args.market_feed_in_price,
    "market_mechanism": args.market_mechanism
}

is_continuous = True

env = MultiMicroGridEnv(env_config, [grid_1, grid_2, grid_3, grid_4])
# 基线对比阶段默认关闭扰动
env.disruption_config['enabled'] = True
policy = MAPPO(num_agents = env.n_agents, obs_dim=env.observation_space['agent_0'].shape[0], action_dim=env.action_space['agent_0'].shape[0], is_continue=is_continuous, actor_lr=args.actor_lr, critic_lr=args.critic_lr, horizon=args.horizon, device=args.device, trick=args.trick)
model_path = ['./model/MAPPO_simple_pay.pth', './model/MAPPO_mrda_pay.pth','./model/MAPPO_msmrda_pay.pth','./model/MAPPO_vda_pay.pth']
model_to_mech = {
    'MAPPO_simple_pay.pth': 'simple',
    'MAPPO_mrda_pay.pth': 'mrda',
    'MAPPO_msmrda_pay.pth': 'msmrda',
    'MAPPO_vda_pay.pth': 'vda'
}

demand_list = []
generation_list = []
common_purchase_list = []
emergency_purchase_list = []
feed_in_list = []
sold_list = []
bought_list = []
storage_list = []
bidding_price_list = []
reward_list = []
net_list = []
storage_list = []


os.makedirs('./data', exist_ok=True)

# ------------------------------------------------------------
# 第1组：无扰动，按不同市场机制的模型进行基线性能对比
# ------------------------------------------------------------
for i in range(len(model_path)):
    # 从参数获取多随机种子；若未提供，则使用默认列表
    seeds = args.seeds if (args.seeds and len(args.seeds) > 0) else [23, 32, 43, 54]

    print(f"Loading model: {model_path[i]}")
    # 使用静态方法创建新的policy实例
    policy = MAPPO.load(num_agents = env.n_agents, obs_dim=env.observation_space['agent_0'].shape[0], action_dim=env.action_space['agent_0'].shape[0], is_continue=is_continuous, model_dir = model_path[i], trick=args.trick)

    # 将policy移动到正确的设备
    if args.device != 'cpu':
        policy.device = args.device
        for agent in policy.agents.values():
            agent.actor = agent.actor.to(args.device)
            agent.device = args.device
        policy.critic = policy.critic.to(args.device)

    # 基线阶段关闭扰动
    env.disruption_config['enabled'] = True

    # 同步环境市场机制到对应模型机制
    policy_basename = os.path.basename(model_path[i])
    mech = model_to_mech.get(policy_basename, args.market_mechanism)
    env.config['market_mechanism'] = mech
    if hasattr(env, 'market_mechanism'):
        env.market_mechanism = mech

    # 按多随机种子重复运行
    for seed in seeds:
        set_seed(seed)
        flag = True
        env.reset()
        while flag:
            if env.current_step % 24 == 0:
                env.day_ahead_dispatch()  
            obs = env.p2p_bidding_preparation()
            actions, log_probs = policy.select_action(obs)
            price, qty_list = env.action_to_bid(actions)
            next_obs, reward, terminated, truncated, info = env.bidding_step(qty_list)
            done = {f'agent_{aid.id}': terminated[f'agent_{aid.id}'] or truncated[f'agent_{aid.id}'] for aid in env.agents}
            policy_tag = os.path.splitext(os.path.basename(model_path[i]))[0]
            for aid in env.agents:
                agent_key = f'agent_{aid.id}'
                step_idx = env.current_step-1

                demand_cur = aid.demand[step_idx % 24] if isinstance(aid.demand, np.ndarray) else aid.demand
                generation_cur = aid.res_generation[step_idx % 24] if isinstance(aid.res_generation, np.ndarray) else aid.res_generation
                common_purchase_cur = aid.common_purchase[step_idx % 24] if isinstance(aid.common_purchase, np.ndarray) else aid.common_purchase

                meta = {'scenario': 'baseline', 'freq_hours': None, 'disruption': False, 'mechanism': mech, 'seed': seed}
                demand_list.append({'policy': policy_tag, 'agent': agent_key, 'step': step_idx, 'value': demand_cur, **meta})
                generation_list.append({'policy': policy_tag, 'agent': agent_key, 'step': step_idx, 'value': generation_cur, **meta})
                common_purchase_list.append({'policy': policy_tag, 'agent': agent_key, 'step': step_idx, 'value': common_purchase_cur, **meta})
                emergency_purchase_list.append({'policy': policy_tag, 'agent': agent_key, 'step': step_idx, 'value': aid.emergency_purchase, **meta})
                feed_in_list.append({'policy': policy_tag, 'agent': agent_key, 'step': step_idx, 'value': aid.feed_in_power, **meta})
                sold_list.append({'policy': policy_tag, 'agent': agent_key, 'step': step_idx, 'value': aid.sold_power, **meta})
                bought_list.append({'policy': policy_tag, 'agent': agent_key, 'step': step_idx, 'value': aid.bought_power, **meta})
                storage_list.append({'policy': policy_tag, 'agent': agent_key, 'step': step_idx, 'value': aid.storage, **meta})
                reward_list.append({'policy': policy_tag, 'agent': agent_key, 'step': step_idx, 'value': reward[agent_key], **meta})
                bidding_price_list.append({'policy': policy_tag, 'agent': agent_key, 'step': step_idx, 'value': aid.price, **meta})
                net_list.append({'policy': policy_tag, 'agent': agent_key, 'step': step_idx, 'value': aid.current_net_power, **meta})

            # 若所有 agent 均结束，则跳出 while 循环并重置环境
            if any(done.values()):
                flag = False
                env.reset()

# ------------------------------------------------------------
# 将收集的数据保存为 CSV
# ------------------------------------------------------------

def _save_metric(records, filename):
    if not records:
        return
    keys = records[0].keys()
    with open(f'./data/{filename}', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)

# 数据保存将在所有实验完成后进行

# ------------------------------------------------------------
# 第2组：鲁棒性验证分两步（1）仅扰动；（2）仅低频交易（2/3/4小时）
# ------------------------------------------------------------
print("开始鲁棒性实验：两阶段（仅扰动；仅低频交易）...")

# 只使用simple_pay模型进行每隔一小时交易实验
model_path_simple_hourly = ['./model/MAPPO_simple_pay.pth']

for i in range(len(model_path_simple_hourly)):
    # 使用与基线实验相同的多随机种子
    seeds_robust = args.seeds if (args.seeds and len(args.seeds) > 0) else [21, 32, 43, 54]
    
    print(f"Loading model for simple hourly trading: {model_path_simple_hourly[i]}")
    # 使用静态方法创建新的policy实例
    policy_simple_hourly = MAPPO.load(num_agents = env.n_agents, obs_dim=env.observation_space['agent_0'].shape[0], action_dim=env.action_space['agent_0'].shape[0], is_continue=is_continuous, model_dir = model_path_simple_hourly[i], trick=args.trick)
    
    # 将policy移动到正确的设备
    if args.device != 'cpu':
        policy_simple_hourly.device = args.device
        for agent in policy_simple_hourly.agents.values():
            agent.actor = agent.actor.to(args.device)
            agent.device = args.device
        policy_simple_hourly.critic = policy_simple_hourly.critic.to(args.device)
    
    # 阶段A：仅扰动（保持每小时P2P交易）
    for seed in seeds_robust:
        set_seed(seed)
        env.disruption_config['enabled'] = True
        env.config['market_mechanism'] = 'simple'
        if hasattr(env, 'market_mechanism'):
            env.market_mechanism = 'simple'

        flag = True
        env.reset()
        while flag:
            if env.current_step % 24 == 0:
                env.day_ahead_dispatch()
            # 每小时都进行P2P
            obs = env.p2p_bidding_preparation()
            actions, log_probs = policy_simple_hourly.select_action(obs)
            price, qty_list = env.action_to_bid(actions)
            next_obs, reward, terminated, truncated, info = env.bidding_step(qty_list)

            done = {f'agent_{aid.id}': terminated[f'agent_{aid.id}'] or truncated[f'agent_{aid.id}'] for aid in env.agents}
            policy_tag = "under_simple_pay_disruption_only"

            for aid in env.agents:
                agent_key = f'agent_{aid.id}'
                step_idx = env.current_step-1

                demand_cur = aid.demand[step_idx % 24] if isinstance(aid.demand, np.ndarray) else aid.demand
                generation_cur = aid.res_generation[step_idx % 24] if isinstance(aid.res_generation, np.ndarray) else aid.res_generation
                common_purchase_cur = aid.common_purchase[step_idx % 24] if isinstance(aid.common_purchase, np.ndarray) else aid.common_purchase

                meta = {'scenario': 'robust_disruption_only', 'freq_hours': 1, 'disruption': True, 'mechanism': 'simple', 'seed': seed}
                demand_list.append({'policy': policy_tag, 'agent': agent_key, 'step': step_idx, 'value': demand_cur, **meta})
                generation_list.append({'policy': policy_tag, 'agent': agent_key, 'step': step_idx, 'value': generation_cur, **meta})
                common_purchase_list.append({'policy': policy_tag, 'agent': agent_key, 'step': step_idx, 'value': common_purchase_cur, **meta})
                emergency_purchase_list.append({'policy': policy_tag, 'agent': agent_key, 'step': step_idx, 'value': aid.emergency_purchase, **meta})
                feed_in_list.append({'policy': policy_tag, 'agent': agent_key, 'step': step_idx, 'value': aid.feed_in_power, **meta})
                sold_list.append({'policy': policy_tag, 'agent': agent_key, 'step': step_idx, 'value': aid.sold_power, **meta})
                bought_list.append({'policy': policy_tag, 'agent': agent_key, 'step': step_idx, 'value': aid.bought_power, **meta})
                storage_list.append({'policy': policy_tag, 'agent': agent_key, 'step': step_idx, 'value': aid.storage, **meta})
                reward_list.append({'policy': policy_tag, 'agent': agent_key, 'step': step_idx, 'value': reward[agent_key], **meta})
                bidding_price_list.append({'policy': policy_tag, 'agent': agent_key, 'step': step_idx, 'value': aid.price, **meta})
                net_list.append({'policy': policy_tag, 'agent': agent_key, 'step': step_idx, 'value': aid.current_net_power, **meta})

            if any(done.values()):
                flag = False
                env.reset()

    # 阶段B：仅低频交易（关闭扰动），分别测试2/3/4小时
    env.disruption_config['enabled'] = True
    for freq_hours in [2, 3, 4]:
        env.config['market_mechanism'] = 'simple'
        if hasattr(env, 'market_mechanism'):
            env.market_mechanism = 'simple'

        for seed in seeds_robust:
            set_seed(seed)
            flag = True
            env.reset()
            while flag:
                if env.current_step % 24 == 0:
                    env.day_ahead_dispatch()
                
                current_hour = env.current_step % 24
                allow_p2p_trading = (current_hour % freq_hours == 0)
                
                if allow_p2p_trading:
                    obs = env.p2p_bidding_preparation()
                    actions, log_probs = policy_simple_hourly.select_action(obs)
                    price, qty_list = env.action_to_bid(actions)
                    next_obs, reward, terminated, truncated, info = env.bidding_step(qty_list)
                else:
                    for agent in env.agents:
                        agent.reset(semi=True)
                        agent.net_calculation(time_step=env.current_step % 24)
                        agent.settle_trade(bought=0, sold=0, total_price=0)
                        current_prices = dict(env.prices)
                        current_prices['emergency_price'] = env.get_emergency_price(env.current_step % 24)
                        agent.update_after_trade(unit_price=current_prices, time_step=env.current_step % 24)

                    reward = {f'agent_{i}': agent.first_reward for i, agent in enumerate(env.agents)}
                    env.current_step += 1
                    terminated = {f'agent_{i}': False for i in range(env.n_agents)}
                    truncated = {f'agent_{i}': (env.current_step >= env.max_steps) for i in range(env.n_agents)}

                done = {f'agent_{aid.id}': terminated[f'agent_{aid.id}'] or truncated[f'agent_{aid.id}'] for aid in env.agents}
                policy_tag = f"under_simple_pay_lowfreq_{freq_hours}h"

                for aid in env.agents:
                    agent_key = f'agent_{aid.id}'
                    step_idx = env.current_step-1

                    demand_cur = aid.demand[step_idx % 24] if isinstance(aid.demand, np.ndarray) else aid.demand
                    generation_cur = aid.res_generation[step_idx % 24] if isinstance(aid.res_generation, np.ndarray) else aid.res_generation
                    common_purchase_cur = aid.common_purchase[step_idx % 24] if isinstance(aid.common_purchase, np.ndarray) else aid.common_purchase

                    meta = {'scenario': 'robust_lowfreq_only', 'freq_hours': freq_hours, 'disruption': False, 'mechanism': 'simple', 'seed': seed}
                    demand_list.append({'policy': policy_tag, 'agent': agent_key, 'step': step_idx, 'value': demand_cur, **meta})
                    generation_list.append({'policy': policy_tag, 'agent': agent_key, 'step': step_idx, 'value': generation_cur, **meta})
                    common_purchase_list.append({'policy': policy_tag, 'agent': agent_key, 'step': step_idx, 'value': common_purchase_cur, **meta})
                    emergency_purchase_list.append({'policy': policy_tag, 'agent': agent_key, 'step': step_idx, 'value': aid.emergency_purchase, **meta})
                    feed_in_list.append({'policy': policy_tag, 'agent': agent_key, 'step': step_idx, 'value': aid.feed_in_power, **meta})
                    sold_list.append({'policy': policy_tag, 'agent': agent_key, 'step': step_idx, 'value': aid.sold_power, **meta})
                    bought_list.append({'policy': policy_tag, 'agent': agent_key, 'step': step_idx, 'value': aid.bought_power, **meta})
                    storage_list.append({'policy': policy_tag, 'agent': agent_key, 'step': step_idx, 'value': aid.storage, **meta})
                    reward_list.append({'policy': policy_tag, 'agent': agent_key, 'step': step_idx, 'value': reward[agent_key], **meta})
                    bidding_price_list.append({'policy': policy_tag, 'agent': agent_key, 'step': step_idx, 'value': aid.price, **meta})
                    net_list.append({'policy': policy_tag, 'agent': agent_key, 'step': step_idx, 'value': aid.current_net_power, **meta})

                if any(done.values()):
                    flag = False
                    env.reset()

print("鲁棒性实验完成！")

# ------------------------------------------------------------
# 多随机种子结果统计汇总
# ------------------------------------------------------------
def aggregate_multi_seed_results(records_list, metric_name):
    """对多随机种子结果进行统计汇总"""
    if not records_list:
        return []
    
    # 按 (policy, agent, step) 分组
    grouped = {}
    for record in records_list:
        key = (record['policy'], record['agent'], record['step'])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(record)
    
    # 计算每个组的统计量
    aggregated_records = []
    for key, group_records in grouped.items():
        values = [r['value'] for r in group_records]
        meta = group_records[0].copy()  # 复制元信息
        del meta['value']  # 删除原始值
        del meta['seed']   # 删除种子信息
        
        # 计算统计量
        aggregated_records.append({
            **meta,
            'value_mean': np.mean(values),
            'value_std': np.std(values),
            'value_min': np.min(values),
            'value_max': np.max(values),
            'value_median': np.median(values),
            'n_seeds': len(values)
        })
    
    return aggregated_records

print("开始多随机种子结果统计汇总...")

# 对每个指标进行统计汇总
demand_aggregated = aggregate_multi_seed_results(demand_list, 'demand')
generation_aggregated = aggregate_multi_seed_results(generation_list, 'generation')
common_purchase_aggregated = aggregate_multi_seed_results(common_purchase_list, 'common_purchase')
emergency_purchase_aggregated = aggregate_multi_seed_results(emergency_purchase_list, 'emergency_purchase')
feed_in_aggregated = aggregate_multi_seed_results(feed_in_list, 'feed_in')
sold_aggregated = aggregate_multi_seed_results(sold_list, 'sold')
bought_aggregated = aggregate_multi_seed_results(bought_list, 'bought')
storage_aggregated = aggregate_multi_seed_results(storage_list, 'storage')
reward_aggregated = aggregate_multi_seed_results(reward_list, 'reward')
bidding_price_aggregated = aggregate_multi_seed_results(bidding_price_list, 'bidding_price')
net_aggregated = aggregate_multi_seed_results(net_list, 'net')

# ------------------------------------------------------------
# 保存原始数据和统计汇总数据
# ------------------------------------------------------------
print("保存所有实验数据...")

# 保存原始数据（包含所有种子的记录）
_save_metric(demand_list, 'demand_records_raw.csv')
_save_metric(generation_list, 'generation_records_raw.csv')
_save_metric(common_purchase_list, 'common_purchase_records_raw.csv')
_save_metric(emergency_purchase_list, 'emergency_purchase_records_raw.csv')
_save_metric(feed_in_list, 'feed_in_records_raw.csv')
_save_metric(sold_list, 'sold_records_raw.csv')
_save_metric(bought_list, 'bought_records_raw.csv')
_save_metric(storage_list, 'storage_records_raw.csv')
_save_metric(reward_list, 'reward_records_raw.csv')
_save_metric(bidding_price_list, 'bidding_price_records_raw.csv')
_save_metric(net_list, 'net_records_raw.csv')

# 保存统计汇总数据（平均值、标准差等）
_save_metric(demand_aggregated, 'demand_records_aggregated.csv')
_save_metric(generation_aggregated, 'generation_records_aggregated.csv')
_save_metric(common_purchase_aggregated, 'common_purchase_records_aggregated.csv')
_save_metric(emergency_purchase_aggregated, 'emergency_purchase_records_aggregated.csv')
_save_metric(feed_in_aggregated, 'feed_in_records_aggregated.csv')
_save_metric(sold_aggregated, 'sold_records_aggregated.csv')
_save_metric(bought_aggregated, 'bought_records_aggregated.csv')
_save_metric(storage_aggregated, 'storage_records_aggregated.csv')
_save_metric(reward_aggregated, 'reward_records_aggregated.csv')
_save_metric(bidding_price_aggregated, 'bidding_price_records_aggregated.csv')
_save_metric(net_aggregated, 'net_records_aggregated.csv')

print("所有实验数据保存完成！")
print("原始数据文件：*_raw.csv")
print("统计汇总数据文件：*_aggregated.csv（包含平均值、标准差等统计量）")

# ------------------------------------------------------------
# 生成Agent级别性能指标统计报告
# ------------------------------------------------------------
def generate_agent_comparison():
    """生成Agent级别的性能指标统计报告"""
    
    # 创建统计目录
    os.makedirs('./statistics', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("开始生成Agent级别性能指标统计...")
    
    # 读取汇总数据
    try:
        reward_df = pd.read_csv('./data/reward_records_aggregated.csv')
        storage_df = pd.read_csv('./data/storage_records_aggregated.csv')
        emergency_df = pd.read_csv('./data/emergency_purchase_records_aggregated.csv')
        feed_in_df = pd.read_csv('./data/feed_in_records_aggregated.csv')
        sold_df = pd.read_csv('./data/sold_records_aggregated.csv')
        bought_df = pd.read_csv('./data/bought_records_aggregated.csv')
        bidding_df = pd.read_csv('./data/bidding_price_records_aggregated.csv')
        net_df = pd.read_csv('./data/net_records_aggregated.csv')
    except FileNotFoundError as e:
        print(f"错误：找不到数据文件 {e}")
        return
    
    # 1. 按policy和agent分组计算指标
    agent_stats = []
    
    for policy in reward_df['policy'].unique():
        policy_reward = reward_df[reward_df['policy'] == policy]
        
        for agent in policy_reward['agent'].unique():
            agent_reward = policy_reward[policy_reward['agent'] == agent]
            agent_storage = storage_df[(storage_df['policy'] == policy) & (storage_df['agent'] == agent)]
            agent_emergency = emergency_df[(emergency_df['policy'] == policy) & (emergency_df['agent'] == agent)]
            agent_feed_in = feed_in_df[(feed_in_df['policy'] == policy) & (feed_in_df['agent'] == agent)]
            agent_sold = sold_df[(sold_df['policy'] == policy) & (sold_df['agent'] == agent)]
            agent_bought = bought_df[(bought_df['policy'] == policy) & (bought_df['agent'] == agent)]
            agent_bidding = bidding_df[(bidding_df['policy'] == policy) & (bidding_df['agent'] == agent)]
            agent_net = net_df[(net_df['policy'] == policy) & (net_df['agent'] == agent)]
            
            # 计算每天的平均指标
            total_days = len(agent_reward) // 24 if len(agent_reward) >= 24 else 1
            
            daily_avg_reward = agent_reward['value_mean'].sum() / total_days
            daily_avg_emergency = agent_emergency['value_mean'].sum() / total_days if len(agent_emergency) > 0 else 0
            daily_avg_feed_in = agent_feed_in['value_mean'].sum() / total_days if len(agent_feed_in) > 0 else 0
            daily_avg_sold = agent_sold['value_mean'].sum() / total_days if len(agent_sold) > 0 else 0
            daily_avg_bought = agent_bought['value_mean'].sum() / total_days if len(agent_bought) > 0 else 0
            avg_storage = agent_storage['value_mean'].mean() if len(agent_storage) > 0 else 0
            avg_bidding_price = agent_bidding['value_mean'].mean() if len(agent_bidding) > 0 else 0
            avg_net = agent_net['value_mean'].mean() if len(agent_net) > 0 else 0
            
            # P2P交易量
            daily_avg_p2p_volume = daily_avg_sold + daily_avg_bought
            
            mechanism = agent_reward['mechanism'].iloc[0] if 'mechanism' in agent_reward.columns and len(agent_reward) > 0 else 'unknown'
            scenario = agent_reward['scenario'].iloc[0] if 'scenario' in agent_reward.columns and len(agent_reward) > 0 else 'unknown'
            
            agent_stats.append({
                'policy': policy,
                'agent': agent,
                'mechanism': mechanism,
                'scenario': scenario,
                'daily_avg_reward': daily_avg_reward,
                'daily_avg_emergency_purchase': daily_avg_emergency,
                'daily_avg_feed_in': daily_avg_feed_in,
                'daily_avg_p2p_volume': daily_avg_p2p_volume,
                'daily_avg_p2p_sold': daily_avg_sold,
                'daily_avg_p2p_bought': daily_avg_bought,
                'avg_storage_level': avg_storage,
                'avg_bidding_price': avg_bidding_price,
                'avg_net_power': avg_net,
                'total_days': total_days
            })
    
    # 保存agent统计
    agent_df = pd.DataFrame(agent_stats)
    agent_df.to_csv(f'./statistics/agent_comparison_{timestamp}.csv', index=False)
    
    # 2. 生成按policy的agent对比报告
    with open(f'./statistics/agent_comparison_by_policy_{timestamp}.txt', 'w', encoding='utf-8') as f:
        f.write("Agent级别性能指标对比报告（按Policy分组）\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*150 + "\n\n")
        
        for policy in agent_df['policy'].unique():
            f.write(f"Policy: {policy}\n")
            f.write("="*150 + "\n")
            
            policy_data = agent_df[agent_df['policy'] == policy]
            
            # 表格格式输出
            f.write(f"{'Agent':<10} {'日均收益':<12} {'日均紧急购买':<15} {'日均上网电量':<15} {'日均售电':<12} {'日均购电':<12} {'日均P2P交易':<15} {'平均储能':<10} {'平均出价':<10} {'净功率':<10}\n")
            f.write("-" * 150 + "\n")
            
            # 按日均收益排序
            policy_sorted = policy_data.sort_values('daily_avg_reward', ascending=False)
            
            for _, row in policy_sorted.iterrows():
                f.write(f"{row['agent']:<10} {row['daily_avg_reward']:<12.2f} {row['daily_avg_emergency_purchase']:<15.2f} {row['daily_avg_feed_in']:<15.2f} "
                       f"{row['daily_avg_p2p_sold']:<12.2f} {row['daily_avg_p2p_bought']:<12.2f} {row['daily_avg_p2p_volume']:<15.2f} "
                       f"{row['avg_storage_level']:<10.3f} {row['avg_bidding_price']:<10.2f} {row['avg_net_power']:<10.2f}\n")
            
            f.write("\n")
            
            # Agent排名
            f.write(f"Agent排名 (按日均收益):\n")
            for i, (_, row) in enumerate(policy_sorted.iterrows(), 1):
                f.write(f"  {i}. {row['agent']}: {row['daily_avg_reward']:.2f}\n")
            
            f.write("\n\n")
    
    # 3. 生成按agent的策略对比报告
    with open(f'./statistics/agent_comparison_by_agent_{timestamp}.txt', 'w', encoding='utf-8') as f:
        f.write("Agent级别性能指标对比报告（按Agent分组）\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*150 + "\n\n")
        
        for agent in agent_df['agent'].unique():
            f.write(f"Agent: {agent}\n")
            f.write("="*150 + "\n")
            
            agent_data = agent_df[agent_df['agent'] == agent]
            
            # 表格格式输出
            f.write(f"{'Policy':<35} {'日均收益':<12} {'日均紧急购买':<15} {'日均上网电量':<15} {'日均售电':<12} {'日均购电':<12} {'日均P2P交易':<15} {'平均储能':<10}\n")
            f.write("-" * 150 + "\n")
            
            # 按日均收益排序
            agent_sorted = agent_data.sort_values('daily_avg_reward', ascending=False)
            
            for _, row in agent_sorted.iterrows():
                f.write(f"{row['policy']:<35} {row['daily_avg_reward']:<12.2f} {row['daily_avg_emergency_purchase']:<15.2f} {row['daily_avg_feed_in']:<15.2f} "
                       f"{row['daily_avg_p2p_sold']:<12.2f} {row['daily_avg_p2p_bought']:<12.2f} {row['daily_avg_p2p_volume']:<15.2f} "
                       f"{row['avg_storage_level']:<10.3f}\n")
            
            f.write("\n")
            
            # Policy排名
            f.write(f"Policy排名 (按日均收益):\n")
            for i, (_, row) in enumerate(agent_sorted.iterrows(), 1):
                f.write(f"  {i}. {row['policy']}: {row['daily_avg_reward']:.2f}\n")
            
            f.write("\n\n")
    
    # 4. 生成综合对比表
    with open(f'./statistics/agent_comparison_summary_{timestamp}.txt', 'w', encoding='utf-8') as f:
        f.write("Agent级别性能指标综合对比报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*150 + "\n\n")
        
        # 整体统计
        f.write("整体统计信息:\n")
        f.write("-" * 80 + "\n")
        f.write(f"总Policy数: {agent_df['policy'].nunique()}\n")
        f.write(f"总Agent数: {agent_df['agent'].nunique()}\n")
        f.write(f"总记录数: {len(agent_df)}\n")
        f.write(f"平均日均收益: {agent_df['daily_avg_reward'].mean():.2f}\n")
        f.write(f"最高日均收益: {agent_df['daily_avg_reward'].max():.2f}\n")
        f.write(f"最低日均收益: {agent_df['daily_avg_reward'].min():.2f}\n")
        f.write(f"收益标准差: {agent_df['daily_avg_reward'].std():.2f}\n\n")
        
        # 按Agent的平均性能
        f.write("按Agent的平均性能 (跨所有Policy):\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Agent':<10} {'平均日均收益':<15} {'平均紧急购买':<15} {'平均上网电量':<15} {'平均P2P交易':<15} {'平均储能':<12}\n")
        f.write("-" * 100 + "\n")
        
        agent_avg = agent_df.groupby('agent').agg({
            'daily_avg_reward': 'mean',
            'daily_avg_emergency_purchase': 'mean',
            'daily_avg_feed_in': 'mean',
            'daily_avg_p2p_volume': 'mean',
            'avg_storage_level': 'mean'
        }).reset_index()
        agent_avg = agent_avg.sort_values('daily_avg_reward', ascending=False)
        
        for _, row in agent_avg.iterrows():
            f.write(f"{row['agent']:<10} {row['daily_avg_reward']:<15.2f} {row['daily_avg_emergency_purchase']:<15.2f} "
                   f"{row['daily_avg_feed_in']:<15.2f} {row['daily_avg_p2p_volume']:<15.2f} {row['avg_storage_level']:<12.3f}\n")
        
        f.write("\n\n")
        
        # 按Policy的平均性能
        f.write("按Policy的平均性能 (跨所有Agent):\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Policy':<35} {'平均日均收益':<15} {'平均紧急购买':<15} {'平均上网电量':<15} {'平均P2P交易':<15} {'平均储能':<12}\n")
        f.write("-" * 100 + "\n")
        
        policy_avg = agent_df.groupby('policy').agg({
            'daily_avg_reward': 'mean',
            'daily_avg_emergency_purchase': 'mean',
            'daily_avg_feed_in': 'mean',
            'daily_avg_p2p_volume': 'mean',
            'avg_storage_level': 'mean'
        }).reset_index()
        policy_avg = policy_avg.sort_values('daily_avg_reward', ascending=False)
        
        for _, row in policy_avg.iterrows():
            f.write(f"{row['policy']:<35} {row['daily_avg_reward']:<15.2f} {row['daily_avg_emergency_purchase']:<15.2f} "
                   f"{row['daily_avg_feed_in']:<15.2f} {row['daily_avg_p2p_volume']:<15.2f} {row['avg_storage_level']:<12.3f}\n")
    
    print(f"Agent级别统计报告已生成:")
    print(f"  - ./statistics/agent_comparison_{timestamp}.csv (CSV数据文件)")
    print(f"  - ./statistics/agent_comparison_by_policy_{timestamp}.txt (按Policy分组)")
    print(f"  - ./statistics/agent_comparison_by_agent_{timestamp}.txt (按Agent分组)")
    print(f"  - ./statistics/agent_comparison_summary_{timestamp}.txt (综合对比)")

# ------------------------------------------------------------
# 生成社区级别性能指标统计报告
# ------------------------------------------------------------
import pandas as pd
from datetime import datetime
import os

def generate_community_statistics():
    """生成社区级别的性能指标统计报告"""
    
    # 创建统计目录
    os.makedirs('./statistics', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("开始生成社区级别性能指标统计...")
    
    # 读取汇总数据
    try:
        reward_df = pd.read_csv('./data/reward_records_aggregated.csv')
        storage_df = pd.read_csv('./data/storage_records_aggregated.csv')
        emergency_df = pd.read_csv('./data/emergency_purchase_records_aggregated.csv')
        feed_in_df = pd.read_csv('./data/feed_in_records_aggregated.csv')
        sold_df = pd.read_csv('./data/sold_records_aggregated.csv')
        bought_df = pd.read_csv('./data/bought_records_aggregated.csv')
        bidding_df = pd.read_csv('./data/bidding_price_records_aggregated.csv')
    except FileNotFoundError as e:
        print(f"错误：找不到数据文件 {e}")
        return
    
    # 1. 社区级别性能对比（按天计算平均值）
    community_stats = []
    
    for policy in reward_df['policy'].unique():
        policy_data = reward_df[reward_df['policy'] == policy]
        
        # 计算每天的平均指标（假设每天24小时）
        # 获取该策略的总天数
        total_days = len(policy_data) // 24 if len(policy_data) >= 24 else 1
        
        # 计算社区级别指标（按天平均）
        daily_avg_reward = policy_data['value_mean'].sum() / total_days
        daily_avg_emergency = emergency_df[emergency_df['policy'] == policy]['value_mean'].sum() / total_days
        daily_avg_feed_in = feed_in_df[feed_in_df['policy'] == policy]['value_mean'].sum() / total_days
        daily_avg_sold = sold_df[sold_df['policy'] == policy]['value_mean'].sum() / total_days
        daily_avg_bought = bought_df[bought_df['policy'] == policy]['value_mean'].sum() / total_days
        avg_storage = storage_df[storage_df['policy'] == policy]['value_mean'].mean()
        avg_bidding_price = bidding_df[bidding_df['policy'] == policy]['value_mean'].mean()
        
        # P2P交易量（按天平均）
        daily_avg_p2p_volume = (daily_avg_sold + daily_avg_bought)
        
        community_stats.append({
            'policy': policy,
            'daily_avg_reward': daily_avg_reward,
            'daily_avg_emergency_purchase': daily_avg_emergency,
            'daily_avg_feed_in': daily_avg_feed_in,
            'daily_avg_p2p_volume': daily_avg_p2p_volume,
            'daily_avg_p2p_sold': daily_avg_sold,
            'daily_avg_p2p_bought': daily_avg_bought,
            'avg_storage_level': avg_storage,
            'avg_bidding_price': avg_bidding_price,
            'total_days': total_days,
            'mechanism': policy_data['mechanism'].iloc[0] if 'mechanism' in policy_data.columns else 'unknown'
        })
    
    # 保存社区统计
    community_df = pd.DataFrame(community_stats)
    community_df.to_csv(f'./statistics/community_comparison_{timestamp}.csv', index=False)
    
    # 2. 生成表格统计报告
    with open(f'./statistics/community_comparison_{timestamp}.txt', 'w', encoding='utf-8') as f:
        f.write("社区级别性能指标对比报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*100 + "\n\n")
        
        # 生成表格格式的统计报告
        f.write("模型性能对比表格 (按天平均):\n")
        f.write("-" * 120 + "\n")
        f.write(f"{'模型名称':<25} {'市场机制':<10} {'日均收益':<12} {'日均紧急购买':<15} {'日均上网电量':<15} {'日均P2P交易':<15} {'平均储能':<10} {'平均出价':<10} {'天数':<6}\n")
        f.write("-" * 120 + "\n")
        
        # 按日均收益排序
        sorted_community = community_df.sort_values('daily_avg_reward', ascending=False)
        
        for _, row in sorted_community.iterrows():
            f.write(f"{row['policy']:<25} {row['mechanism']:<10} {row['daily_avg_reward']:<12.2f} {row['daily_avg_emergency_purchase']:<15.2f} {row['daily_avg_feed_in']:<15.2f} {row['daily_avg_p2p_volume']:<15.2f} {row['avg_storage_level']:<10.3f} {row['avg_bidding_price']:<10.2f} {row['total_days']:<6}\n")
        
        f.write("-" * 100 + "\n\n")
        
        # 按机制分组的详细统计
        f.write("按市场机制分组的详细统计:\n")
        f.write("="*100 + "\n\n")
        
        mechanisms = community_df['mechanism'].unique()
        
        for mechanism in mechanisms:
            f.write(f"市场机制: {mechanism}\n")
            f.write("-" * 50 + "\n")
            
            mech_data = community_df[community_df['mechanism'] == mechanism]
            
            # 计算该机制的统计量（按天平均）
            daily_avg_reward = mech_data['daily_avg_reward'].mean()
            daily_avg_emergency = mech_data['daily_avg_emergency_purchase'].mean()
            daily_avg_feed_in = mech_data['daily_avg_feed_in'].mean()
            daily_avg_p2p = mech_data['daily_avg_p2p_volume'].mean()
            avg_storage = mech_data['avg_storage_level'].mean()
            avg_bidding = mech_data['avg_bidding_price'].mean()
            avg_days = mech_data['total_days'].mean()
            
            f.write(f"  日均收益: {daily_avg_reward:.2f}\n")
            f.write(f"  日均紧急购买: {daily_avg_emergency:.2f}\n")
            f.write(f"  日均上网电量: {daily_avg_feed_in:.2f}\n")
            f.write(f"  日均P2P交易: {daily_avg_p2p:.2f}\n")
            f.write(f"  平均储能水平: {avg_storage:.3f}\n")
            f.write(f"  平均出价: {avg_bidding:.2f}\n")
            f.write(f"  平均天数: {avg_days:.1f}\n")
            f.write(f"  模型数量: {len(mech_data)}\n\n")
            
            # 该机制下的模型排名
            f.write(f"  {mechanism} 机制模型排名 (按日均收益):\n")
            mech_sorted = mech_data.sort_values('daily_avg_reward', ascending=False)
            for i, (_, row) in enumerate(mech_sorted.iterrows(), 1):
                f.write(f"    {i}. {row['policy']}: {row['daily_avg_reward']:.2f}\n")
            f.write("\n")
        
        # 整体性能排名
        f.write("整体性能排名 (按日均收益):\n")
        f.write("="*60 + "\n")
        for i, (_, row) in enumerate(sorted_community.iterrows(), 1):
            f.write(f"{i:2d}. {row['policy']:<25} {row['mechanism']:<10} {row['daily_avg_reward']:>10.2f}\n")
        
        f.write("\n")
        
        # 关键指标汇总
        f.write("关键指标汇总:\n")
        f.write("="*60 + "\n")
        f.write(f"参与测试的模型总数: {len(community_df)}\n")
        f.write(f"市场机制种类: {len(mechanisms)}\n")
        f.write(f"最佳模型: {sorted_community.iloc[0]['policy']} (日均收益: {sorted_community.iloc[0]['daily_avg_reward']:.2f})\n")
        f.write(f"最差模型: {sorted_community.iloc[-1]['policy']} (日均收益: {sorted_community.iloc[-1]['daily_avg_reward']:.2f})\n")
        f.write(f"日均收益差距: {sorted_community.iloc[0]['daily_avg_reward'] - sorted_community.iloc[-1]['daily_avg_reward']:.2f}\n")
        f.write(f"平均测试天数: {community_df['total_days'].mean():.1f}\n")
    
    
    # 4. 鲁棒性分析
    robust_stats = []
    robust_policies = [p for p in community_df['policy'].unique() if 'robust' in p.lower() or 'disruption' in p.lower()]
    
    if robust_policies:
        with open(f'./statistics/robust_community_{timestamp}.txt', 'w', encoding='utf-8') as f:
            f.write("鲁棒性分析报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            
            for policy in robust_policies:
                policy_data = community_df[community_df['policy'] == policy]
                if not policy_data.empty:
                    row = policy_data.iloc[0]
                    f.write(f"策略: {policy}\n")
                    f.write(f"  日均收益: {row['daily_avg_reward']:.2f}\n")
                    f.write(f"  日均紧急购买: {row['daily_avg_emergency_purchase']:.2f}\n")
                    f.write(f"  日均P2P交易量: {row['daily_avg_p2p_volume']:.2f}\n")
                    f.write(f"  平均储能: {row['avg_storage_level']:.3f}\n")
                    f.write(f"  测试天数: {row['total_days']}\n")
                    f.write("\n")
    
    print(f"社区级别统计报告已生成:")
    print(f"  - ./statistics/community_comparison_{timestamp}.txt (表格统计报告)")
    print(f"  - ./statistics/community_comparison_{timestamp}.csv (CSV数据文件)")
    if robust_policies:
        print(f"  - ./statistics/robust_community_{timestamp}.txt (鲁棒性分析)")

# 生成社区级别统计
generate_community_statistics()

# 生成Agent级别统计
generate_agent_comparison()












