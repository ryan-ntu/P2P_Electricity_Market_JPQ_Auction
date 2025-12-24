from module import micro_grid_agent
from environment import MultiMicroGridEnv
from algorithm.MAPPO import MAPPO
from algorithm.MAPPO_i import MAPPO_i
from algorithm.IPPO import IPPO
from algorithm.MADDPG import MADDPG
import numpy as np
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import tqdm
from datetime import datetime
from utils import init_training_csv

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train', help='train or test')
parser.add_argument('--train_mode', type=str, default='from_scratch', choices=['from_scratch', 'continue'], help='训练模式: from_scratch(从零开始) 或 continue(继续训练)')
parser.add_argument('--model_dir', type=str, default='./model', help='模型保存路径')
parser.add_argument('--pretrained_model', type=str, default='./model/MAPPO_simple_pay.pth', help='预训练模型路径(仅在continue模式下使用)')

# 使用示例:
# 从零开始训练: python main.py --train_mode from_scratch
# 继续训练: python main.py --train_mode continue --pretrained_model ./model/MAPPO_simple_pay.pth
# 如果要运行MADDPG 请使用main_for_DDPG.py 这里的DDPG调用有误
parser.add_argument('--load_file', type=str, default='./Dataset/load_profiles.npy', help='负载数据')
parser.add_argument('--generation_file', type=str, default='./Dataset/generation_profiles.npy', help='发电数据')
parser.add_argument("--rl_algorithm", type=str, default='MAPPO', help='所使用的算法名称', choices=['MAPPO', 'IPPO','MADDPG'])
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
parser.add_argument('--grid4_charge', type=float, default=10, help='第4个grid的充电限制')
parser.add_argument('--grid4_discharge', type=float, default=10, help='第4个grid的放电限制')
parser.add_argument('--grid4_storage', type=float, default=20, help='第4个grid的初始储能')

parser.add_argument('--market_common_price', type=float, default=10, help='市场电价')
parser.add_argument('--market_emergency_price', type=float, default=20, help='市场紧急电价')
parser.add_argument('--market_feed_in_price', type=float, default=2, help='市场售电价')
parser.add_argument('--market_max_steps', type=int, default=720, help='市场最大步数')
parser.add_argument('--market_mechanism', type=str, default='simple', help='市场机制')


# 共有参数
parser.add_argument("--device", type=str, default='cuda:0')
parser.add_argument("--seed", type=int, default=100) # 0 10 100  
parser.add_argument("--max_episodes", type=int, default=int(7000))
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
parser.add_argument("--horizon", type=int, default=1024) #
parser.add_argument("--clip_param", type=float, default=0.2)
parser.add_argument("--K_epochs", type=int, default=10) # 15 # 困难任务建议设置为5
parser.add_argument("--entropy_coefficient", type=float, default=0.01)
parser.add_argument("--minibatch_size", type=int, default=512)
parser.add_argument("--lmbda", type=float, default=0.95) # GAE参数
## mappo 参数
parser.add_argument("--huber_delta", type=float, default=10.0) # huber_loss参数
parser.add_argument("--use_shared_critic", type=bool, default=False)
parser.add_argument("--trick", type=dict, default={'adv_norm':True,
                                                        'ObsNorm':False,
                                                        'Batch_ObsNorm':False,
                                                        'reward_norm':False,'reward_scaling':False,    # or
                                                        'orthogonal_init':True,'adam_eps':False,'lr_decay':False, # 原代码中设置为False
                                                        # 以上均在PPO_with_tricks.py中实现过
                                                       'ValueClip':False,'huber_loss':True,
                                                       'LayerNorm':True,'feature_norm':True,
                                                       }) 



args = parser.parse_args()
# 保存用户指定的pretrained_model路径
user_specified_pretrained_model = args.pretrained_model

# 如果用户没有显式指定pretrained_model，则使用默认路径
# 如果用户指定了pretrained_model，则使用用户指定的路径
if args.pretrained_model == './model/MAPPO_simple_pay.pth':  # 这是默认值
    args.pretrained_model = args.model_dir + '/' + 'MAPPO_'  + args.market_mechanism + '_pay.pth'
print(f"使用预训练模型: {args.pretrained_model}")

current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = os.path.join('runs', current_time)
writer = SummaryWriter(log_dir) # 用于记录训练过程中的指标

if args.market_mechanism == 'qty_price':
    action_space = 3
elif args.market_mechanism:
    action_space = 3
else:
    raise ValueError(f"Market mechanism is not supported: {args.market_mechanism}")

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

grid_4 = micro_grid_agent(param_panel4)  # 创建了四个微电网的智能体
  


env_config = {
    "max_steps": args.market_max_steps,
    "common_price": args.market_common_price,
    "emergency_price": args.market_emergency_price,
    "feed_in_price": args.market_feed_in_price,
    "market_mechanism": args.market_mechanism
}

episode_num = 0
is_continuous = True

env = MultiMicroGridEnv(env_config, [grid_1, grid_2, grid_3, grid_4])

# 根据训练模式创建或加载policy
if args.train_mode == 'continue':
    # 使用用户指定的预训练模型路径，而不是被修改后的路径
    pretrained_model_path = user_specified_pretrained_model
    print(f"继续训练模式：从 {pretrained_model_path} 加载预训练模型")
    
    policy = MAPPO.load(num_agents = env.n_agents, obs_dim=env.observation_space['agent_0'].shape[0], action_dim=env.action_space['agent_0'].shape[0], is_continue=is_continuous, model_dir = pretrained_model_path, trick=args.trick, horizon=args.horizon, device=args.device)

    
    policy.actor_lr = args.actor_lr
    policy.critic_lr = args.critic_lr
    policy.horizon = args.horizon

    
    for agent in policy.agents.values():
        if args.trick['adam_eps']:
            agent.actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=args.actor_lr, eps=1e-5)
        else:
            agent.actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=args.actor_lr)

    if args.trick['adam_eps']:
        policy.critic_optimizer = torch.optim.Adam(policy.critic.parameters(), lr=args.critic_lr, eps=1e-5)
    else:
        policy.critic_optimizer = torch.optim.Adam(policy.critic.parameters(), lr=args.critic_lr)

else:
    print(f"从零开始训练模式：创建新的{args.rl_algorithm}模型")
    # 从零开始训练
    if args.rl_algorithm == 'MAPPO':
        if args.use_shared_critic:
            print(f"使用共享critic的MAPPO算法")
            policy = MAPPO(num_agents = env.n_agents, obs_dim=env.observation_space['agent_0'].shape[0], action_dim=env.action_space['agent_0'].shape[0], is_continue=is_continuous, actor_lr=args.actor_lr, critic_lr=args.critic_lr, horizon=args.horizon, device=args.device, trick=args.trick)
        else:
            print(f"使用个体critic的MAPPO算法")
            policy = MAPPO_i(num_agents = env.n_agents, obs_dim=env.observation_space['agent_0'].shape[0], action_dim=env.action_space['agent_0'].shape[0], is_continue=is_continuous, actor_lr=args.actor_lr, critic_lr=args.critic_lr, horizon=args.horizon, device=args.device, trick=args.trick)
    elif args.rl_algorithm == 'IPPO':
        policy = IPPO(num_agents = env.n_agents, obs_dim=env.observation_space['agent_0'].shape[0], action_dim=env.action_space['agent_0'].shape[0], is_continue=is_continuous, actor_lr=args.actor_lr, critic_lr=args.critic_lr, horizon=args.horizon, device=args.device, trick=args.trick)
    elif args.rl_algorithm == 'MADDPG':
        policy = MADDPG(num_agents = env.n_agents, obs_dim=env.observation_space['agent_0'].shape[0], action_dim=env.action_space['agent_0'].shape[0], is_continue=is_continuous, actor_lr=args.actor_lr, critic_lr=args.critic_lr, horizon=args.horizon, device=args.device, trick=args.trick)
    else:
        raise ValueError(f"RL algorithm is not supported: {args.rl_algorithm}")
    # 构建训练体

# 验证模型设备设置
print(f"模型设备设置: {args.device}")
if args.device != 'cpu':
    # 检查所有模型是否在正确的设备上
    for agent_id, agent in policy.agents.items(): # 四个agents
        actor_device = next(agent.actor.parameters()).device
        if actor_device != torch.device(args.device):
            print(f"警告: {agent_id} actor在 {actor_device}，需要移动到 {args.device}")
            agent.actor = agent.actor.to(args.device)
        
        if agent.critic is not None: # 如果critic不为None则也将其放在cuda上
            critic_device = next(agent.critic.parameters()).device
            if critic_device != torch.device(args.device):
                print(f"警告: {agent_id} critic在 {critic_device}，需要移动到 {args.device}")
                agent.critic = agent.critic.to(args.device)


    if hasattr(policy, 'critic'):
        critic_device = next(policy.critic.parameters()).device
        if critic_device != torch.device(args.device):
            print(f"警告: critic在 {critic_device}，需要移动到 {args.device}")
            policy.critic = policy.critic.to(args.device)

if args.use_shared_critic:
    csv_writer, csv_f, csv_path = init_training_csv(env.n_agents, filename = "training_log.csv",algorithm_name=args.rl_algorithm,market_mechanism=args.market_mechanism)
else:
    csv_writer, csv_f, csv_path = init_training_csv(env.n_agents, filename = "training_log_individual_critic.csv",algorithm_name=args.rl_algorithm,market_mechanism=args.market_mechanism)

env.reset() # 重置环境

# 输出算法相关参数
print("=" * 60)
print("算法训练参数配置:")
print("=" * 60)
print(f"算法名称: {args.rl_algorithm}")
print(f"Horizon: {args.horizon}")
print(f"K_epochs: {args.K_epochs}")
print(f"Use Shared Critic: {args.use_shared_critic}")
print(f"Clip Param: {args.clip_param}")
print(f"Entropy Coefficient: {args.entropy_coefficient}")
print(f"Actor LR: {args.actor_lr}")
print(f"Critic LR: {args.critic_lr}")
print(f"Huber Delta: {args.huber_delta}")
print("\nTrick 配置:")
for key, value in args.trick.items():
    print(f"  {key}: {value}")
print("=" * 60)
print()

episode_reward = {f'agent_{i}': 0 for i in range(env.n_agents)} # 初始化每个agent的奖励
episode_emergency_purchase = {f'agent_{i}': 0 for i in range(env.n_agents)} # 初始化每个agent的紧急购买电量
episode_feed_in_power = {f'agent_{i}': 0 for i in range(env.n_agents)} # 初始化每个agent的售电电量
episode_storage_level = {f'agent_{i}': 0 for i in range(env.n_agents)} # 初始化每个agent的储能电量
episode_cost = {f'agent_{i}': 0 for i in range(env.n_agents)} # 初始化每个agent的成本
# 添加进度条
pbar = tqdm.tqdm(total=args.max_episodes, desc='Training Progress')

while episode_num < args.max_episodes: # 训练代码 这里的max episode 是5000
    if env.current_step % 24 == 0: # 24h生成一次预测数据和购电计划
        env.day_ahead_dispatch() # 为每个agent 生成当天的预测数据并给出次日24h购电计划

    obs = env.p2p_bidding_preparation() # 计算当前小时每个agent 的需要或者富裕多少功率 并生成观测
    actions, log_probs = policy.select_action(obs) # agent的动作向量 以及每个动作维度的对数概率 用于学习
    
    price, qty_list = env.action_to_bid(actions) # 将策略输出的动作转换成竞价价格 和竞价数量 

    next_obs, reward, terminated, truncated, info = env.bidding_step(qty_list) # 撮合推进一步环境交互 并返回下一个观测

    done = {f'agent_{aid.id}': terminated[f'agent_{aid.id}'] or truncated[f'agent_{aid.id}'] for aid in env.agents} #是否结束或者触发截断
    adv_done = {f'agent_{aid.id}': terminated[f'agent_{aid.id}'] for aid in env.agents} # 用于GAE截断

    policy.add(obs, actions, reward, next_obs, done, log_probs, adv_done) # 添加到经验回放中

    ## Update and record
    episode_reward = {f'agent_{i}': episode_reward[f'agent_{i}'] + reward[f'agent_{i}'] for i in range(env.n_agents)}# 累计每个agent的奖励
    episode_emergency_purchase = {f'agent_{i}': episode_emergency_purchase[f'agent_{i}']+ env.agents[i].emergency_purchase for i in range(env.n_agents)}# 累计每个agent的紧急购买电量
    episode_feed_in_power = {f'agent_{i}': episode_feed_in_power[f'agent_{i}'] + env.agents[i].feed_in_power for i in range(env.n_agents)} #倒卖到电量的电量
    episode_storage_level = {f'agent_{i}': episode_storage_level[f'agent_{i}'] + env.agents[i].storage/env.agents[i].parameter_battery for i in range(env.n_agents)} # 电池当前电量占电池容量的比例
    episode_cost = {f'agent_{i}': episode_cost[f'agent_{i}'] + env.agents[i].first_reward for i in range(env.n_agents)} # 每个电网在实时竞价阶段的收益不包括日前公共购电的成本

        

    if (episode_num*env.max_steps+env.current_step) % args.horizon == 0: # 每隔horizon步触发一次集中式学习 每个agent 的策略都会各自更新 价值网络是共享的 并且用全体agent的观测一起更新
        if args.rl_algorithm == 'MAPPO' or args.rl_algorithm == 'IPPO':
            policy.learn(minibatch_size=args.minibatch_size,
                        gamma=args.gamma,
                        lmbda=args.lmbda,
                        clip_param=args.clip_param,
                        K_epochs=args.K_epochs,
                        entropy_coefficient=args.entropy_coefficient,
                        huber_delta=args.huber_delta)
            # policy.learn 内部会 reset buffer
            if args.trick['lr_decay']:
                policy.lr_decay(episode_num, args.max_episodes) # 学习率衰减
        elif args.rl_algorithm == 'MADDPG':
            policy.learn(batch_size=args.minibatch_size,
                        gamma=args.gamma,
                        tau=args.tau)

    
    
    if any(done.values()): # 用于tensor board记录
        normalized = {'reward': {
            f'agent_{i}': episode_reward[f'agent_{i}'] / env.max_steps
            for i in range(env.n_agents)
        }, 'emergency_purchase': {
            f'agent_{i}': episode_emergency_purchase[f'agent_{i}'] / env.max_steps
            for i in range(env.n_agents)
        }, 'feed_in_power': {
            f'agent_{i}': episode_feed_in_power[f'agent_{i}'] / env.max_steps
            for i in range(env.n_agents)
        }, 'storage_level': {
            f'agent_{i}': episode_storage_level[f'agent_{i}'] / env.max_steps
        for i in range(env.n_agents)
        }, 'cost': {
            f'agent_{i}': episode_cost[f'agent_{i}'] / env.max_steps
            for i in range(env.n_agents)
        }
        }
        
        if episode_num % 50 == 0:
            # writer.add_scalars('Agent/Reward', normalized['reward'], episode_num)
            # writer.add_scalars('Agent/EmergencyPurchase', normalized['emergency_purchase'], episode_num)
            # writer.add_scalars('Agent/FeedInPower', normalized['feed_in_power'], episode_num)
            # writer.add_scalars('Agent/StorageLevel', normalized['storage_level'], episode_num)
            # writer.add_scalars('Agent/Cost', normalized['cost'], episode_num)


            total_reward = sum(normalized['reward'].values())
            total_emergency = sum(normalized['emergency_purchase'].values())
            total_feed_in = sum(normalized['feed_in_power'].values())
            total_cost = sum(normalized['cost'].values())
            # writer.add_scalar('Total/Reward', total_reward, episode_num)
            # writer.add_scalar('Total/EmergencyPurchase', total_emergency, episode_num)
            # writer.add_scalar('Total/FeedInPower', total_feed_in, episode_num)
            # writer.add_scalar('Total/Cost', total_cost, episode_num)

            row = [episode_num] # 写入对应的内容到csv文件中
            for i in range(env.n_agents):
                row.extend([
                    normalized['reward'][f'agent_{i}'],
                    normalized['emergency_purchase'][f'agent_{i}'],
                    normalized['feed_in_power'][f'agent_{i}'],
                    normalized['storage_level'][f'agent_{i}'],
                    normalized['cost'][f'agent_{i}'],
                ])
            row.extend([total_reward, total_emergency, total_feed_in, total_cost])

            csv_writer.writerow(row)
            csv_f.flush()

        # 4) 递增 episode_num 并打印
        episode_num += 1
        pbar.update(1)  # 更新进度条
        pbar.set_postfix({
            'Episode': episode_num,
            'Step': env.current_step
        })

        # 5) 重置环境并把各-agent 指标清零
        env.reset()
        zero_dict = {f'agent_{i}': 0 for i in range(env.n_agents)}
        episode_reward = zero_dict.copy()
        episode_emergency_purchase = zero_dict.copy()
        episode_feed_in_power = zero_dict.copy()
        episode_storage_level = zero_dict.copy()
        episode_cost = zero_dict.copy()


pbar.close()  # 关闭进度条
writer.close()
policy.save(model_dir=args.model_dir,name=env.market_mechanism)

