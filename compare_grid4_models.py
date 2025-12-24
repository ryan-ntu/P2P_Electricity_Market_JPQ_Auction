#!/usr/bin/env python3
"""
对比测试不同发电和存储能力组合的模型性能
基于example_intra_day.py的结构，测试grid4_test_*目录下的所有模型
记录系统的总体收益、交易量、紧急能源购买、feed_in和平均存储水平
"""

from module import micro_grid_agent
from environment import MultiMicroGridEnv
from algorithm.MAPPO import MAPPO

import numpy as np 
import argparse
import os, csv
import torch
import random
import glob
import sys
from datetime import datetime

# 设置所有随机种子以确保结果可重现
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# 在参数解析之前设置种子
set_seed(10)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='test', help='test mode')
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
parser.add_argument("--seed", type=int, default=100)
parser.add_argument("--horizon", type=int, default=720)
parser.add_argument("--trick", type=dict, default={'adv_norm':False,
                                                    'ObsNorm':False,
                                                    'reward_norm':False,'reward_scaling':False,
                                                    'orthogonal_init':True,'adam_eps':False,'lr_decay':False,
                                                    'ValueClip':False,'huber_loss':False,
                                                    'LayerNorm':True,'feature_norm':False,
                                                    }) 

args = parser.parse_args()

action_space = 2

def create_grid_agents(grid4_generation, grid4_battery):
    """根据grid4参数创建智能体"""
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
        "generation_param": grid4_generation,  # 使用传入的参数
        "battery_lim": grid4_battery,  # 使用传入的参数
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
    
    return [grid_1, grid_2, grid_3, grid_4]

def run_single_test(model_path, grid4_generation, grid4_battery, test_name):
    """运行单个模型测试"""
    print(f"\n{'='*80}")
    print(f"测试模型: {test_name}")
    print(f"Grid4 发电能力: {grid4_generation}, 电池容量: {grid4_battery}")
    print(f"模型路径: {model_path}")
    print(f"{'='*80}")
    
    # 创建环境
    agents = create_grid_agents(grid4_generation, grid4_battery)
    env_config = {
        "max_steps": args.market_max_steps,
        "common_price": args.market_common_price,
        "emergency_price": args.market_emergency_price,
        "feed_in_price": args.market_feed_in_price,
        "market_mechanism": args.market_mechanism
    }
    
    env = MultiMicroGridEnv(env_config, agents)
    env.disruption_config['enabled'] = False  # 关闭扰动
    
    # 加载模型
    is_continuous = True
    policy = MAPPO.load(
        num_agents=env.n_agents, 
        obs_dim=env.observation_space['agent_0'].shape[0], 
        action_dim=env.action_space['agent_0'].shape[0], 
        is_continue=is_continuous, 
        model_dir=model_path, 
        trick=args.trick,
        horizon=args.horizon,
        device=args.device
    )
    
    # 将policy移动到正确的设备
    if args.device != 'cpu':
        policy.device = args.device
        for agent in policy.agents.values():
            agent.actor = agent.actor.to(args.device)
            agent.device = args.device
        policy.critic = policy.critic.to(args.device)
    
    # 运行测试 - 记录每小时数据
    env.reset()
    
    # 初始化记录数据结构
    hourly_data = []  # 存储每小时的数据
    hourly_stats = {f'agent_{i}': {
        'reward': [[] for _ in range(24)],  # 24小时，每小时一个列表
        'emergency_purchase': [[] for _ in range(24)],
        'feed_in_power': [[] for _ in range(24)],
        'bought_power': [[] for _ in range(24)],  # P2P购电量
        'sold_power': [[] for _ in range(24)],    # P2P售电量
        'community_cost': [[] for _ in range(24)],  # P2P市场成交总额
        'storage_level': [[] for _ in range(24)],
        'bidding_price': [[] for _ in range(24)],
        'bidding_qty': [[] for _ in range(24)]  # 竞价数量
    } for i in range(env.n_agents)}
    
    step_count = 0
    
    while step_count < env.max_steps:
        if env.current_step % 24 == 0:
            env.day_ahead_dispatch()
        
        obs = env.p2p_bidding_preparation()
        actions, log_probs = policy.select_action(obs)
        price, qty_list = env.action_to_bid(actions)
        next_obs, reward, terminated, truncated, info = env.bidding_step(qty_list)
        
        # 保存qty_list用于记录
        current_qty_list = qty_list.copy()
        
        done = {f'agent_{aid.id}': terminated[f'agent_{aid.id}'] or truncated[f'agent_{aid.id}'] for aid in env.agents}
        
        # 计算当前天数和小时
        current_day = step_count // 24 + 1
        current_hour = step_count % 24
        
        # 记录每小时数据
        hour_data = {
            'test_name': test_name,
            'grid4_generation': grid4_generation,
            'grid4_battery': grid4_battery,
            'hour': step_count,
            'day': current_day,
            'agents': {}
        }
        
        for i, agent in enumerate(env.agents):
            agent_key = f'agent_{i}'
            storage_level = agent.storage / agent.parameter_battery
            
            # 计算P2P市场成交总额（统计买卖双方的community_cost）
            # 对于agent级别，统计所有交易的community_cost
            if hasattr(agent, 'community_cost'):
                community_cost = agent.community_cost
            else:
                community_cost = 0
            
            # 获取bidding_price (agent对象使用price属性)
            bidding_price = agent.price if hasattr(agent, 'price') else 0
            
            # 获取bidding_qty (从qty_list中获取)
            bidding_qty = current_qty_list[i] if i < len(current_qty_list) else 0
            
            # 记录每小时数据
            hour_data['agents'][agent_key] = {
                'reward': reward[agent_key],
                'emergency_purchase': agent.emergency_purchase,
                'feed_in_power': agent.feed_in_power,
                'bought_power': agent.bought_power,
                'sold_power': agent.sold_power,
                'community_cost': community_cost,
                'storage_level': storage_level,
                'bidding_price': bidding_price,
                'bidding_qty': bidding_qty
            }
            
            # 按小时收集数据（跨所有天）
            hourly_stats[agent_key]['reward'][current_hour].append(reward[agent_key])
            hourly_stats[agent_key]['emergency_purchase'][current_hour].append(agent.emergency_purchase)
            hourly_stats[agent_key]['feed_in_power'][current_hour].append(agent.feed_in_power)
            hourly_stats[agent_key]['bought_power'][current_hour].append(agent.bought_power)
            hourly_stats[agent_key]['sold_power'][current_hour].append(agent.sold_power)
            hourly_stats[agent_key]['community_cost'][current_hour].append(community_cost)
            hourly_stats[agent_key]['storage_level'][current_hour].append(storage_level)
            hourly_stats[agent_key]['bidding_price'][current_hour].append(bidding_price)
            hourly_stats[agent_key]['bidding_qty'][current_hour].append(bidding_qty)
        
        hourly_data.append(hour_data)
        step_count += 1
        
        if any(done.values()):
            break
    
    # 计算各时刻的平均值（跨所有天）
    daily_averages = {}
    
    for agent_key in [f'agent_{i}' for i in range(env.n_agents)]:
        # 计算每个小时的平均值（跨所有天）
        hourly_avg_reward = []
        hourly_avg_emergency = []
        hourly_avg_feed_in = []
        hourly_avg_bought = []
        hourly_avg_sold = []
        hourly_avg_community_cost = []
        hourly_avg_storage = []
        hourly_avg_bidding_price = []
        hourly_avg_bidding_qty = []
        
        for hour in range(24):
            if hourly_stats[agent_key]['reward'][hour]:  # 如果该小时有数据
                hourly_avg_reward.append(np.mean(hourly_stats[agent_key]['reward'][hour]))
                hourly_avg_emergency.append(np.mean(hourly_stats[agent_key]['emergency_purchase'][hour]))
                hourly_avg_feed_in.append(np.mean(hourly_stats[agent_key]['feed_in_power'][hour]))
                hourly_avg_bought.append(np.mean(hourly_stats[agent_key]['bought_power'][hour]))
                hourly_avg_sold.append(np.mean(hourly_stats[agent_key]['sold_power'][hour]))
                hourly_avg_community_cost.append(np.mean(hourly_stats[agent_key]['community_cost'][hour]))
                hourly_avg_storage.append(np.mean(hourly_stats[agent_key]['storage_level'][hour]))
                hourly_avg_bidding_price.append(np.mean(hourly_stats[agent_key]['bidding_price'][hour]))
                hourly_avg_bidding_qty.append(np.mean(hourly_stats[agent_key]['bidding_qty'][hour]))
        
        # 计算总平均值（所有时刻的平均）
        if hourly_avg_reward:
            daily_averages[agent_key] = {
                'avg_reward_per_hour': np.mean(hourly_avg_reward),
                'avg_emergency_per_hour': np.mean(hourly_avg_emergency),
                'avg_feed_in_per_hour': np.mean(hourly_avg_feed_in),
                'avg_bought_power_per_hour': np.mean(hourly_avg_bought),
                'avg_sold_power_per_hour': np.mean(hourly_avg_sold),
                'avg_community_cost_per_hour': np.mean(hourly_avg_community_cost),
                'avg_storage_level': np.mean(hourly_avg_storage),
                'avg_bidding_price_per_hour': np.mean(hourly_avg_bidding_price),
                'avg_bidding_qty_per_hour': np.mean(hourly_avg_bidding_qty),
                'total_reward': sum(sum(hourly_stats[agent_key]['reward'][hour]) for hour in range(24)),
                'total_emergency_purchase': sum(sum(hourly_stats[agent_key]['emergency_purchase'][hour]) for hour in range(24)),
                'total_feed_in_power': sum(sum(hourly_stats[agent_key]['feed_in_power'][hour]) for hour in range(24)),
                'total_bought_power': sum(sum(hourly_stats[agent_key]['bought_power'][hour]) for hour in range(24)),
                'total_sold_power': sum(sum(hourly_stats[agent_key]['sold_power'][hour]) for hour in range(24)),
                'total_community_cost': sum(sum(hourly_stats[agent_key]['community_cost'][hour]) for hour in range(24)),
                'total_bidding_qty': sum(sum(hourly_stats[agent_key]['bidding_qty'][hour]) for hour in range(24))
            }
    
    # 计算系统总体指标
    # 系统总成交额需要重新计算，只统计卖家的成交额（正值）
    system_total_community_cost = 0
    for agent_key in [f'agent_{i}' for i in range(env.n_agents)]:
        if agent_key in daily_averages:
            # 只统计卖家的成交额（total_community_cost > 0表示净收入）
            agent_community_cost = daily_averages[agent_key]['total_community_cost']
            if agent_community_cost > 0:  # 只有净收入（卖家）才统计成交额
                system_total_community_cost += agent_community_cost
    
    system_totals = {
        'total_reward': sum(agent_data['total_reward'] for agent_data in daily_averages.values()),
        'total_emergency_purchase': sum(agent_data['total_emergency_purchase'] for agent_data in daily_averages.values()),
        'total_feed_in_power': sum(agent_data['total_feed_in_power'] for agent_data in daily_averages.values()),
        'total_p2p_volume': sum(agent_data['total_bought_power'] for agent_data in daily_averages.values()),  # 系统总P2P成交量
        'total_community_cost': system_total_community_cost,  # 只统计卖家的成交额
        'avg_storage_level': sum(agent_data['avg_storage_level'] for agent_data in daily_averages.values()) / len(daily_averages)
    }
    
    results = {
        'test_name': test_name,
        'grid4_generation': grid4_generation,
        'grid4_battery': grid4_battery,
        'model_path': model_path,
        'total_steps': step_count,
        'total_days': step_count // 24,
        'system_totals': system_totals,
        'daily_averages': daily_averages,
        'hourly_data': hourly_data,
        'test_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    print(f"测试完成:")
    print(f"  总步数: {step_count} (总天数: {step_count // 24})")
    print(f"  系统总收益: {system_totals['total_reward']:.2f}")
    print(f"  系统总紧急购买: {system_totals['total_emergency_purchase']:.2f}")
    print(f"  系统总FIT: {system_totals['total_feed_in_power']:.2f}")
    print(f"  系统总P2P成交量: {system_totals['total_p2p_volume']:.2f}")
    print(f"  系统总P2P市场成交额: {system_totals['total_community_cost']:.2f}")
    print(f"  系统平均存储水平: {system_totals['avg_storage_level']:.3f}")
    
    return results

class TeeOutput:
    """同时输出到控制台和文件的类"""
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

def main():
    """主函数"""
    # 创建输出文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"grid4_battery_comparison_gen15_{timestamp}.txt"
    
    # 重定向输出到文件和控制台
    tee = TeeOutput(output_file)
    sys.stdout = tee
    
    try:
        print("开始Grid4模型对比测试")
        print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"对比目标: 发电能力为15，不同电池容量的模型")
        print(f"结果将同时保存到: {output_file}")
        print("="*80)
        
        # 查找所有grid4_test_*目录
        model_dirs = glob.glob(os.path.join(args.model_dir, "grid4_test_*"))
        model_dirs = [d for d in model_dirs if os.path.isdir(d)]
        
        if not model_dirs:
            print("❌ 未找到任何grid4_test_*目录")
            return
        
        print(f"找到 {len(model_dirs)} 个模型目录:")
        for model_dir in model_dirs:
            print(f"  - {model_dir}")
        
        # 解析模型参数 - 仅选择发电能力为15的模型
        model_configs = []
        target_generation = 15.0  # 目标发电能力
        
        for model_dir in model_dirs:
            model_name = os.path.basename(model_dir)
            model_path = os.path.join(model_dir, "MAPPO_simple_pay.pth")
            
            if not os.path.exists(model_path):
                print(f"⚠️  跳过 {model_name}: 模型文件不存在")
                continue
            
            # 解析参数: grid4_test_gen_X_bat_Y
            try:
                parts = model_name.split('_')
                if len(parts) >= 6 and parts[0] == 'grid4' and parts[1] == 'test':
                    generation = float(parts[3])  # gen_X
                    battery = float(parts[5])     # bat_Y
                    
                    # 只选择发电能力为15的模型
                    if generation == target_generation:
                        model_configs.append({
                            'name': model_name,
                            'path': model_path,
                            'generation': generation,
                            'battery': battery
                        })
                        print(f"✅ 选择模型: {model_name} (发电={generation}, 电池={battery})")
                    else:
                        print(f"⏭️  跳过 {model_name}: 发电能力 {generation} 不等于目标值 {target_generation}")
                else:
                    print(f"⚠️  跳过 {model_name}: 无法解析参数")
            except (ValueError, IndexError) as e:
                print(f"⚠️  跳过 {model_name}: 参数解析错误 - {e}")
        
        if not model_configs:
            print("❌ 没有找到有效的模型配置")
            return
        
        print(f"\n将测试 {len(model_configs)} 个模型 (发电能力均为15):")
        for config in model_configs:
            print(f"  - {config['name']}: 发电={config['generation']}, 电池={config['battery']}")
        
        # 运行所有测试
        all_results = []
        for i, config in enumerate(model_configs, 1):
            print(f"\n进度: {i}/{len(model_configs)}")
            try:
                result = run_single_test(
                    config['path'], 
                    config['generation'], 
                    config['battery'], 
                    config['name']
                )
                all_results.append(result)
            except Exception as e:
                print(f"❌ 测试 {config['name']} 失败: {e}")
                all_results.append({
                    'test_name': config['name'],
                    'grid4_generation': config['generation'],
                    'grid4_battery': config['battery'],
                    'model_path': config['path'],
                    'error': str(e),
                    'test_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
    
        # 保存结果
        # 保存详细结果到CSV
        summary_file = f"grid4_battery_summary_gen15_{timestamp}.csv"
        hourly_file = f"grid4_battery_hourly_gen15_{timestamp}.csv"
        
        if all_results:
            # 按电池容量排序所有结果
            sorted_results = sorted([r for r in all_results if 'error' not in r], 
                                    key=lambda x: x['grid4_battery'])
            
            # 创建汇总表格数据
            summary_data = []
            hourly_csv_data = []
        
            for result in sorted_results:
                if 'error' not in result:
                    # 系统级汇总数据
                    summary_row = {
                        'test_name': result['test_name'],
                        'grid4_generation': result['grid4_generation'],
                        'grid4_battery': result['grid4_battery'],
                        'total_days': result['total_days'],
                        'system_total_reward': result['system_totals']['total_reward'],
                        'system_total_emergency': result['system_totals']['total_emergency_purchase'],
                        'system_total_feed_in': result['system_totals']['total_feed_in_power'],
                        'system_total_p2p_volume': result['system_totals']['total_p2p_volume'],
                        'system_total_community_cost': result['system_totals']['total_community_cost'],
                        'system_avg_storage_level': result['system_totals']['avg_storage_level']
                    }
                    
                    # 添加每个agent的每日平均数据
                    for agent_key, agent_data in result['daily_averages'].items():
                        agent_id = agent_key.split('_')[1]
                        summary_row[f'agent_{agent_id}_avg_reward_per_hour'] = agent_data['avg_reward_per_hour']
                        summary_row[f'agent_{agent_id}_avg_emergency_per_hour'] = agent_data['avg_emergency_per_hour']
                        summary_row[f'agent_{agent_id}_avg_feed_in_per_hour'] = agent_data['avg_feed_in_per_hour']
                        summary_row[f'agent_{agent_id}_avg_bought_power_per_hour'] = agent_data['avg_bought_power_per_hour']
                        summary_row[f'agent_{agent_id}_avg_sold_power_per_hour'] = agent_data['avg_sold_power_per_hour']
                        summary_row[f'agent_{agent_id}_avg_community_cost_per_hour'] = agent_data['avg_community_cost_per_hour']
                        summary_row[f'agent_{agent_id}_avg_storage_level'] = agent_data['avg_storage_level']
                        summary_row[f'agent_{agent_id}_avg_bidding_price_per_hour'] = agent_data['avg_bidding_price_per_hour']
                        summary_row[f'agent_{agent_id}_total_reward'] = agent_data['total_reward']
                        summary_row[f'agent_{agent_id}_total_emergency'] = agent_data['total_emergency_purchase']
                        summary_row[f'agent_{agent_id}_total_feed_in'] = agent_data['total_feed_in_power']
                        summary_row[f'agent_{agent_id}_total_bought_power'] = agent_data['total_bought_power']
                        summary_row[f'agent_{agent_id}_total_sold_power'] = agent_data['total_sold_power']
                        summary_row[f'agent_{agent_id}_total_community_cost'] = agent_data['total_community_cost']
                    
                    summary_data.append(summary_row)
                    
                    # 创建每小时数据的CSV格式
                    for hour_data in result['hourly_data']:
                        for agent_key, agent_data in hour_data['agents'].items():
                            hourly_row = {
                                'test_name': hour_data['test_name'],
                                'grid4_generation': hour_data['grid4_generation'],
                                'grid4_battery': hour_data['grid4_battery'],
                                'day': hour_data['day'],
                                'hour': hour_data['hour'],
                                'agent': agent_key,
                                'reward': agent_data['reward'],
                                'emergency_purchase': agent_data['emergency_purchase'],
                                'feed_in_power': agent_data['feed_in_power'],
                                'bought_power': agent_data['bought_power'],
                                'sold_power': agent_data['sold_power'],
                                'community_cost': agent_data['community_cost'],
                                'storage_level': agent_data['storage_level'],
                                'bidding_price': agent_data['bidding_price'],
                                'bidding_qty': agent_data['bidding_qty']
                            }
                            hourly_csv_data.append(hourly_row)
        
            # 保存汇总数据
            if summary_data:
                summary_keys = summary_data[0].keys()
                with open(summary_file, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=summary_keys)
                    writer.writeheader()
                    writer.writerows(summary_data)
            
            # 保存每小时数据
            if hourly_csv_data:
                hourly_keys = hourly_csv_data[0].keys()
                with open(hourly_file, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=hourly_keys)
                    writer.writeheader()
                    writer.writerows(hourly_csv_data)
        
            print(f"\n{'='*80}")
            print("测试结果总结 - 发电能力15，电池容量对比")
            print(f"{'='*80}")
            
            # 创建对比表格
            print("\n📊 系统总体性能对比表 (发电能力固定为15，按电池容量排序):")
            print("-" * 150)
            print(f"{'模型名称':<25} {'发电':<6} {'电池':<6} {'总收益':<10} {'总紧急购买':<12} {'总FIT':<10} {'总P2P成交量':<12} {'总P2P成交额':<12} {'平均存储':<10}")
            print("-" * 150)
            
            for result in sorted_results:
                print(f"{result['test_name']:<25} {result['grid4_generation']:<6} {result['grid4_battery']:<6} "
                      f"{result['system_totals']['total_reward']:<10.2f} {result['system_totals']['total_emergency_purchase']:<12.2f} "
                      f"{result['system_totals']['total_feed_in_power']:<10.2f} {result['system_totals']['total_p2p_volume']:<12.2f} "
                      f"{result['system_totals']['total_community_cost']:<12.2f} {result['system_totals']['avg_storage_level']:<10.3f}")
            
            # 按电池容量分组显示结果 (发电能力固定为15)
            print("\n📈 电池容量对比结果 (发电能力固定为15，按电池容量排序):")
            
            print(f"{'电池容量':<8} {'总收益':<10} {'紧急购买':<10} {'FIT':<8} {'P2P成交量':<12} {'P2P成交额':<12} {'存储水平':<10} {'模型名称':<25}")
            print("-" * 105)
            for result in sorted_results:
                print(f"{result['grid4_battery']:<8} {result['system_totals']['total_reward']:<10.2f} "
                      f"{result['system_totals']['total_emergency_purchase']:<10.2f} {result['system_totals']['total_feed_in_power']:<8.2f} "
                      f"{result['system_totals']['total_p2p_volume']:<12.2f} {result['system_totals']['total_community_cost']:<12.2f} "
                      f"{result['system_totals']['avg_storage_level']:<10.3f} {result['test_name']:<25}")
            
            # 分析电池容量的影响
            print(f"\n🔍 电池容量影响分析:")
            if len(sorted_results) > 1:
                min_bat_result = sorted_results[0]
                max_bat_result = sorted_results[-1]
                print(f"  最小电池容量: {min_bat_result['grid4_battery']} -> 总收益: {min_bat_result['system_totals']['total_reward']:.2f}")
                print(f"  最大电池容量: {max_bat_result['grid4_battery']} -> 总收益: {max_bat_result['system_totals']['total_reward']:.2f}")
                print(f"  收益差异: {max_bat_result['system_totals']['total_reward'] - min_bat_result['system_totals']['total_reward']:.2f}")
                
                # 找出最佳配置
                best_result = max(sorted_results, key=lambda x: x['system_totals']['total_reward'])
                print(f"  最佳配置: 电池容量 {best_result['grid4_battery']}, 总收益: {best_result['system_totals']['total_reward']:.2f}")
            
            # 显示每个agent的详细指标 (按电池容量排序)
            print("\n👥 各Agent每日平均指标 (按电池容量排序):")
            print("-" * 150)
            print(f"{'模型':<20} {'Agent':<8} {'每小时收益':<10} {'每小时紧急':<10} {'每小时FIT':<10} {'每小时购电':<10} {'每小时售电':<10} {'每小时成交额':<12} {'存储水平':<10} {'竞价价格':<10} {'竞价数量':<10}")
            print("-" * 150)
            
            for result in sorted_results:
                for agent_key, agent_data in result['daily_averages'].items():
                    print(f"{result['test_name']:<20} {agent_key:<8} {agent_data['avg_reward_per_hour']:<10.3f} "
                          f"{agent_data['avg_emergency_per_hour']:<10.3f} {agent_data['avg_feed_in_per_hour']:<10.3f} "
                          f"{agent_data['avg_bought_power_per_hour']:<10.3f} {agent_data['avg_sold_power_per_hour']:<10.3f} "
                          f"{agent_data['avg_community_cost_per_hour']:<12.3f} {agent_data['avg_storage_level']:<10.3f} "
                          f"{agent_data['avg_bidding_price_per_hour']:<10.3f} {agent_data['avg_bidding_qty_per_hour']:<10.3f}")
            
            print(f"\n📁 文件保存:")
            print(f"  汇总数据: {summary_file}")
            print(f"  每小时数据: {hourly_file}")
            print(f"  完整结果报告: {output_file}")
            print(f"测试完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("❌ 没有成功的测试结果")
    
    finally:
        # 恢复标准输出并关闭文件
        sys.stdout = tee.terminal
        tee.close()
        print(f"\n✅ 测试完成！完整结果已保存到: {output_file}")

if __name__ == "__main__":
    main()
