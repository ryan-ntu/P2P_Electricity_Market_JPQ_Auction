import pandas as pd
import numpy as np
import os
from datetime import datetime

# ---- 配置区域 ----
DATA_DIR = './data'
NUM_HOURS = 24  # first day hours
agents = ['agent_0', 'agent_1', 'agent_2', 'agent_3']
# 仅用于第一组基线对比：只保留四个基线策略
policies = ['MAPPO_simple_pay', 'MAPPO_mrda_pay', 'MAPPO_msmrda_pay', 'MAPPO_vda_pay']

# 文件映射
files = {
    'demand': 'demand_records.csv',
    'sold': 'sold_records.csv',
    'feed_in': 'feed_in_records.csv',
    'generation': 'generation_records.csv',
    'common_purchase': 'common_purchase_records.csv',
    'bought': 'bought_records.csv',
    'emergency': 'emergency_purchase_records.csv',
    'reward': 'reward_records.csv',
    'net': 'net_records.csv',
    'storage': 'storage_records.csv',
    'bidding_price': 'bidding_price_records.csv'
}

# 检查文件存在
for fname in files.values():
    path = os.path.join(DATA_DIR, fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到文件: {path}")

# 读取所有数据并添加 day/hour 信息
data = {}
for comp, fname in files.items():
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    df['day'] = df['step'] // 24
    df['hour'] = df['step'] % 24
    
    # # 将 payoff 值除以 100 以便比较
    # if comp == 'cost':
    #     df['value'] = df['value'] / 100
    
    data[comp] = df  # 只保留第 0 天

# 1) 仅用于组织，可按需扩展
load_comps = ['demand', 'sold', 'feed_in']
supply_comps = ['generation', 'common_purchase', 'bought', 'emergency']

# 创建输出目录
if not os.path.exists('statistics'):
    os.makedirs('statistics')

# 基线阶段需要的指标（含 bidding_price）
metrics = [
    ('sold', 'Total Sold (kWh)'),
    ('bought', 'Total Bought (kWh)'),
    ('reward', 'Total Reward'),
    ('emergency', 'Total Emergency Purchase (kWh)'),
    ('feed_in', 'Total Feed-in (kWh)'),
    ('net', 'Global Net Power (kWh)'),
    ('storage', 'Total Storage (kWh)'),
    ('bidding_price', 'Unit Bidding Price ($)')
]

# 创建统计结果文件
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = f"statistics/agent_statistics_{timestamp}.txt"

with open(output_file, 'w', encoding='utf-8') as f:
    f.write("Agent Statistics Report\n")
    f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*80 + "\n\n")
    
    for metric, ylabel in metrics:
        f.write(f"\n{ylabel} Statistics\n")
        f.write("-" * 50 + "\n")
        
        df_metric = data[metric]

        # 每小时的平均（跨所有天）
        hourly_mean = (
            df_metric.groupby(['policy', 'agent', 'hour'], as_index=False)['value']
            .mean()
            .rename(columns={'value': 'hourly_avg'})
        )

        # 每天总量的平均（先对每一天求和，再跨天平均）
        daily_total_mean = (
            df_metric.groupby(['policy', 'agent', 'day'], as_index=False)['value']
            .sum()
            .groupby(['policy', 'agent'], as_index=False)['value']
            .mean()
            .rename(columns={'value': 'daily_total_avg'})
        )

        # 输出每小时统计数据
        f.write("\nHourly Average (across all days):\n")
        f.write("Policy\t\tAgent\t\tHour\t\tAverage\n")
        f.write("-" * 50 + "\n")
        
        for policy in policies:
            for agent in agents:
                agent_policy_hours = hourly_mean[(hourly_mean['agent'] == agent) & (hourly_mean['policy'] == policy)]
                if not agent_policy_hours.empty:
                    for _, row in agent_policy_hours.iterrows():
                        f.write(f"{policy:<15}\t{agent:<10}\t{int(row['hour']):<5}\t\t{row['hourly_avg']:.4f}\n")
        
        # 输出每日总量统计
        f.write("\nDaily Total Average (across all days):\n")
        f.write("Policy\t\tAgent\t\tDaily Total Average\n")
        f.write("-" * 50 + "\n")
        
        for policy in policies:
            for agent in agents:
                agent_daily = daily_total_mean[(daily_total_mean['agent'] == agent) & (daily_total_mean['policy'] == policy)]
                if not agent_daily.empty:
                    avg_val = agent_daily['daily_total_avg'].iloc[0]
                    f.write(f"{policy:<15}\t{agent:<10}\t{avg_val:.4f}\n")
        
        f.write("\n" + "="*80 + "\n")

print(f"Statistics saved to: {output_file}")

# -------------- 额外统计：所有 agent 的跨指标对比 --------------
selected_metrics = ['sold', 'bought', 'emergency', 'feed_in', 'net', 'reward']

# 创建跨指标对比统计文件
comparison_file = f"statistics/agent_comparison_{timestamp}.txt"

with open(comparison_file, 'w', encoding='utf-8') as f:
    f.write("Agent Cross-Metrics Comparison Report\n")
    f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*80 + "\n\n")
    
    f.write("Daily Total Average across metrics:\n")
    f.write("Agent\t\tPolicy\t\t\t")
    for metric in selected_metrics:
        f.write(f"{metric:<12}")
    f.write("\n")
    f.write("-" * 80 + "\n")
    
    for agent in agents:
        for policy in policies:
            f.write(f"{agent:<10}\t{policy:<15}\t")
            for metric in selected_metrics:
                df_m = data[metric]
                agent_daily_mean = (
                    df_m[df_m['agent'] == agent]
                    .groupby(['policy', 'day'], as_index=False)['value']
                    .sum()
                )
                agent_daily_mean = (
                    agent_daily_mean[agent_daily_mean['policy'] == policy]
                    .groupby('policy', as_index=False)['value']
                    .mean()
                )
                val = float(agent_daily_mean['value'].iloc[0]) if not agent_daily_mean.empty else 0.0
                f.write(f"{val:<12.4f}")
            f.write("\n")
        f.write("\n")

print(f"Cross-metrics comparison saved to: {comparison_file}")

# -------------- 社区整体（4 个 agent 汇总）跨指标对比统计 --------------
community_file = f"statistics/community_comparison_{timestamp}.txt"

with open(community_file, 'w', encoding='utf-8') as f:
    f.write("Community Cross-Metrics Comparison Report\n")
    f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*80 + "\n\n")
    
    f.write("Community Daily Total Average across metrics:\n")
    f.write("Policy\t\t\t")
    for metric in selected_metrics:
        f.write(f"{metric:<12}")
    f.write("\n")
    f.write("-" * 80 + "\n")
    
    for policy in policies:
        f.write(f"{policy:<15}\t")
        for metric in selected_metrics:
            df_m = data[metric]
            # 社区层面：按 day 汇总所有 agent 再取均值
            community_daily = (
                df_m.groupby(['policy', 'day'], as_index=False)['value']
                .sum()
            )
            community_daily = (
                community_daily[community_daily['policy'] == policy]
                .groupby('policy', as_index=False)['value']
                .mean()
            )
            val = float(community_daily['value'].iloc[0]) if not community_daily.empty else 0.0
            f.write(f"{val:<12.4f}")
        f.write("\n")

print(f"Community comparison saved to: {community_file}")

# ============================================================
# 鲁棒性统计（不与其他基线比较；不画 bidding_price）
# 
# 我们使用 CSV 中新增的字段：scenario、freq_hours、disruption、mechanism
# 场景一：robust_disruption_only（仅扰动、每小时P2P）
# 场景二：robust_lowfreq_only（仅低频：2/3/4小时）
# ============================================================

# 过滤出可用的鲁棒场景标签
available_scenarios = set()
for comp, df in data.items():
    if 'scenario' in df.columns:
        available_scenarios.update(df['scenario'].dropna().unique().tolist())

robust_metrics = [
    ('sold', 'Total Sold (kWh)'),
    ('bought', 'Total Bought (kWh)'),
    ('reward', 'Total Reward'),
    ('emergency', 'Total Emergency Purchase (kWh)'),
    ('feed_in', 'Total Feed-in (kWh)'),
    ('net', 'Global Net Power (kWh)'),
    ('storage', 'Total Storage (kWh)'),
    ('bidding_price', 'Unit Bidding Price ($)')
]

# -------------- 鲁棒性：Agent-level 与 Community-level 跨指标统计 --------------
def generate_robust_statistics():
    # 指标列表（含 reward，不含 bidding_price 在柱状里可选，这里仍保留）
    bar_metrics = ['sold', 'bought', 'reward', 'emergency', 'feed_in', 'net', 'storage']
    categories = [
        ('baseline_simple', ('baseline', 'MAPPO_simple_pay', None)),
        ('disruption_only', ('robust_disruption_only', None, None)),
        ('lowfreq_2h', ('robust_lowfreq_only', None, 2)),
        ('lowfreq_3h', ('robust_lowfreq_only', None, 3)),
        ('lowfreq_4h', ('robust_lowfreq_only', None, 4)),
    ]

    # Agent-level 鲁棒性统计
    robust_agent_file = f"statistics/robust_agents_{timestamp}.txt"
    
    with open(robust_agent_file, 'w', encoding='utf-8') as f:
        f.write("Robust Agents Statistics Report\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write("Daily Total Average across metrics:\n")
        f.write("Agent\t\tCategory\t\t\t")
        for metric in bar_metrics:
            f.write(f"{metric:<12}")
        f.write("\n")
        f.write("-" * 100 + "\n")
        
        for agent in agents:
            for c_name, (scenario, policy_name, freq_h) in categories:
                f.write(f"{agent:<10}\t{c_name:<20}\t")
                for metric in bar_metrics:
                    df_m = data[metric]
                    df_sel = df_m.copy()
                    if scenario:
                        df_sel = df_sel[df_sel.get('scenario', '') == scenario]
                    if policy_name:
                        df_sel = df_sel[df_sel['policy'] == policy_name]
                    if freq_h is not None and 'freq_hours' in df_sel.columns:
                        df_sel = df_sel[df_sel['freq_hours'] == freq_h]
                    df_sel = df_sel[df_sel['agent'] == agent]
                    val = 0.0
                    if not df_sel.empty:
                        daily_totals = df_sel.groupby(['day'], as_index=False)['value'].sum()
                        if not daily_totals.empty:
                            val = float(daily_totals['value'].mean())
                    f.write(f"{val:<12.4f}")
                f.write("\n")
            f.write("\n")

    print(f"Robust agents statistics saved to: {robust_agent_file}")

    # Community-level 鲁棒性统计
    robust_community_file = f"statistics/robust_community_{timestamp}.txt"
    
    with open(robust_community_file, 'w', encoding='utf-8') as f:
        f.write("Robust Community Statistics Report\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write("Community Daily Total Average across metrics:\n")
        f.write("Category\t\t\t")
        for metric in bar_metrics:
            f.write(f"{metric:<12}")
        f.write("\n")
        f.write("-" * 100 + "\n")
        
        for c_name, (scenario, policy_name, freq_h) in categories:
            f.write(f"{c_name:<20}\t")
            for metric in bar_metrics:
                df_m = data[metric]
                df_sel = df_m.copy()
                if scenario:
                    df_sel = df_sel[df_sel.get('scenario', '') == scenario]
                if policy_name:
                    df_sel = df_sel[df_sel['policy'] == policy_name]
                if freq_h is not None and 'freq_hours' in df_sel.columns:
                    df_sel = df_sel[df_sel['freq_hours'] == freq_h]
                val = 0.0
                if not df_sel.empty:
                    community_daily = df_sel.groupby(['day'], as_index=False)['value'].sum()
                    if not community_daily.empty:
                        val = float(community_daily['value'].mean())
                f.write(f"{val:<12.4f}")
            f.write("\n")

    print(f"Robust community statistics saved to: {robust_community_file}")

generate_robust_statistics()

# -------------- 鲁棒性：合并每小时统计（baseline 1h simple + disruption-only + lowfreq 2/3/4h） --------------
def generate_robust_hourly_statistics():
    has_dis = 'robust_disruption_only' in available_scenarios
    has_lf = 'robust_lowfreq_only' in available_scenarios
    if not (has_dis or has_lf):
        return
    
    robust_hourly_file = f"statistics/robust_hourly_{timestamp}.txt"
    
    with open(robust_hourly_file, 'w', encoding='utf-8') as f:
        f.write("Robust Hourly Statistics Report\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        for metric, ylabel in robust_metrics:
            f.write(f"\n{ylabel} Hourly Statistics\n")
            f.write("-" * 50 + "\n")
            
            df_metric = data[metric]
            # baseline 1h simple
            df_base = df_metric[(df_metric.get('scenario', '') == 'baseline') & (df_metric['policy'] == 'MAPPO_simple_pay')]
            hourly_base = (
                df_base.groupby(['policy', 'agent', 'hour'], as_index=False)['value']
                .mean()
                .rename(columns={'value': 'hourly_avg'})
            ) if not df_base.empty else pd.DataFrame(columns=['policy','agent','hour','hourly_avg'])

            # disruption-only
            hourly_dis = pd.DataFrame(columns=['policy','agent','hour','hourly_avg'])
            if has_dis:
                df_dis = df_metric[df_metric.get('scenario', '') == 'robust_disruption_only']
                if not df_dis.empty:
                    hourly_dis = (
                        df_dis.groupby(['policy', 'agent', 'hour'], as_index=False)['value']
                        .mean()
                        .rename(columns={'value': 'hourly_avg'})
                    )

            # lowfreq-only (2/3/4)
            hourly_lf = pd.DataFrame(columns=['freq_hours','agent','hour','hourly_avg'])
            if has_lf:
                df_lf = df_metric[df_metric.get('scenario', '') == 'robust_lowfreq_only']
                if not df_lf.empty:
                    hourly_lf = (
                        df_lf.groupby(['freq_hours', 'agent', 'hour'], as_index=False)['value']
                        .mean()
                        .rename(columns={'value': 'hourly_avg'})
                    )

            # 输出每小时统计数据
            f.write("\nBaseline (1h simple):\n")
            f.write("Policy\t\tAgent\t\tHour\t\tAverage\n")
            f.write("-" * 50 + "\n")
            for _, row in hourly_base.iterrows():
                f.write(f"{row['policy']:<15}\t{row['agent']:<10}\t{int(row['hour']):<5}\t\t{row['hourly_avg']:.4f}\n")
            
            if has_dis:
                f.write("\nDisruption Only:\n")
                f.write("Policy\t\tAgent\t\tHour\t\tAverage\n")
                f.write("-" * 50 + "\n")
                for _, row in hourly_dis.iterrows():
                    f.write(f"{row['policy']:<15}\t{row['agent']:<10}\t{int(row['hour']):<5}\t\t{row['hourly_avg']:.4f}\n")
            
            if has_lf:
                f.write("\nLow Frequency:\n")
                f.write("Freq_Hours\tAgent\t\tHour\t\tAverage\n")
                f.write("-" * 50 + "\n")
                for _, row in hourly_lf.iterrows():
                    f.write(f"{int(row['freq_hours']):<10}\t{row['agent']:<10}\t{int(row['hour']):<5}\t\t{row['hourly_avg']:.4f}\n")
            
            f.write("\n" + "="*80 + "\n")

    print(f"Robust hourly statistics saved to: {robust_hourly_file}")

generate_robust_hourly_statistics()
