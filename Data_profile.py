import pandas as pd
import numpy as np
import os

# ==============================
# 若已有 .npy 文件则直接加载
# 否则从 CSV 生成后再保存并绘图
# ==============================
data_dir = './Dataset'
load_npy_path = os.path.join(data_dir, 'load_profiles.npy')
gen_npy_path = os.path.join(data_dir, 'generation_profiles.npy')
ids_npy_path = os.path.join(data_dir, 'customer_ids.npy')

if os.path.exists(load_npy_path) and os.path.exists(gen_npy_path) and os.path.exists(ids_npy_path):
    # 直接从已有的 .npy 文件加载
    print("检测到已有 .npy 配置文件，直接加载用于绘图。")
    load_profiles_array = np.load(load_npy_path)
    generation_profiles_array = np.load(gen_npy_path)
    customer_ids_list = np.load(ids_npy_path).tolist()

    print(f"   - 负载配置文件: {load_npy_path} (形状: {load_profiles_array.shape})")
    print(f"   - 发电配置文件: {gen_npy_path} (形状: {generation_profiles_array.shape})")
    print(f"   - 客户ID: {ids_npy_path} (客户: {customer_ids_list})")

else:
    # 若没有 .npy 文件，则从 CSV 生成
    print("未检测到 .npy 配置文件，从 CSV 重新生成并保存。")

    # 正确读取CSV文件，跳过第一行描述，使用第二行作为头部
    df = pd.read_csv('./Dataset/2012-2013 Solar home electricity data v2.csv', skiprows=1)

    # 随机选择4个客户
    index = np.random.choice(range(1, 300), size=4, replace=False)

    # 筛选2012年7月的数据，并转换日期格式
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    july_2012 = df[(df['date'] >= '2012-07-01') & (df['date'] <= '2012-07-31')]

    # 获取半小时时间列（从0:30到0:00的48列）
    time_columns = [col for col in df.columns if ':' in col]
    print(f"时间列数量: {len(time_columns)}")
    print(f"前5个时间列: {time_columns[:5]}")

    # 显示选中的客户索引
    print(f"随机选择的客户: {index}")
    print(f"7月数据形状: {july_2012.shape}")
    print(f"数据类别: {july_2012['Consumption Category'].unique()}")

    print("\n" + "="*50)
    print("生成标准化负载和发电配置文件")
    print("="*50)

    # 为每个选中的客户生成配置文件
    for i, customer_id in enumerate(index):
        print(f"\n客户 {customer_id}:")
        
        # 提取该客户的数据
        customer_data = july_2012[july_2012['Customer'] == customer_id]
        
        if customer_data.empty:
            print(f"  警告: 客户 {customer_id} 在7月没有数据")
            continue
        
        # 提取负载数据 (GC) 和发电数据 (GG)
        load_data = customer_data[customer_data['Consumption Category'] == 'GC']
        generation_data = customer_data[customer_data['Consumption Category'] == 'GG']
        
        print(f"  负载数据行数: {len(load_data)}")
        print(f"  发电数据行数: {len(generation_data)}")
        
        if not load_data.empty and not generation_data.empty:
            # 计算7月整月的平均数据
            # 对所有天的48个半小时数据取平均
            load_monthly_avg = load_data[time_columns].astype(float).mean(axis=0).values
            generation_monthly_avg = generation_data[time_columns].astype(float).mean(axis=0).values
            
            # 将48个半小时数据合并为24个小时数据（每两个半小时取平均）
            hourly_load = []
            hourly_generation = []
            
            for hour in range(24):
                # 每小时包含两个半小时段：hour*2 和 hour*2+1
                idx1, idx2 = hour * 2, hour * 2 + 1
                hourly_load.append((load_monthly_avg[idx1] + load_monthly_avg[idx2]) / 2)
                hourly_generation.append((generation_monthly_avg[idx1] + generation_monthly_avg[idx2]) / 2)
            
            hourly_load = np.array(hourly_load)
            hourly_generation = np.array(hourly_generation)
            
            # 标准化到[0,1]范围
            load_normalized = (hourly_load - hourly_load.min()) / (hourly_load.max() - hourly_load.min()) if hourly_load.max() > hourly_load.min() else hourly_load
            generation_normalized = (hourly_generation - hourly_generation.min()) / (hourly_generation.max() - hourly_generation.min()) if hourly_generation.max() > hourly_generation.min() else hourly_generation
            
            print(f"  使用数据: 7月整月平均 ({len(load_data)}天)")
            print(f"  负载配置文件 (前8个小时值): {load_normalized[:8].round(3)}")
            print(f"  发电配置文件 (前8个小时值): {generation_normalized[:8].round(3)}")
            print(f"  负载范围: {hourly_load.min():.3f} - {hourly_load.max():.3f}")
            print(f"  发电范围: {hourly_generation.min():.3f} - {hourly_generation.max():.3f}")
            
            # 可以将配置文件存储起来供后续使用
            globals()[f'customer_{customer_id}_load_profile'] = load_normalized
            globals()[f'customer_{customer_id}_generation_profile'] = generation_normalized

    print("\n" + "="*50)
    print("完整的24小时配置文件示例（客户 {})".format(index[0]))
    print("="*50)

    # 创建24小时时间标签
    hour_labels = [f"{hour:02d}:00" for hour in range(24)]

    # 显示第一个客户的完整配置文件
    customer_id = index[0]
    load_profile = globals()[f'customer_{customer_id}_load_profile']
    generation_profile = globals()[f'customer_{customer_id}_generation_profile']

    print("小时     负载    发电")
    print("-" * 25)
    for i, hour_label in enumerate(hour_labels):
        print(f"{hour_label:>6}   {load_profile[i]:.3f}   {generation_profile[i]:.3f}")
        
    # 将完整的配置文件数据输出为便于查看的格式
    load_values_clean = [round(float(x), 3) for x in load_profile]
    generation_values_clean = [round(float(x), 3) for x in generation_profile]
    print(f"\n完整负载配置文件 (24个小时值): {load_values_clean}")
    print(f"完整发电配置文件 (24个小时值): {generation_values_clean}")

    print("\n配置文件统计:")
    print(f"负载 - 最小值: {load_profile.min():.3f}, 最大值: {load_profile.max():.3f}, 平均值: {load_profile.mean():.3f}")
    print(f"发电 - 最小值: {generation_profile.min():.3f}, 最大值: {generation_profile.max():.3f}, 平均值: {generation_profile.mean():.3f}")

    # 识别发电高峰时段（发电量 > 0.5）
    peak_generation_hours = [hour_labels[i] for i in range(len(generation_profile)) if generation_profile[i] > 0.5]
    print(f"\n发电高峰时段 (>0.5): {peak_generation_hours}")

    # 识别负载高峰时段（负载 > 0.7）
    peak_load_hours = [hour_labels[i] for i in range(len(load_profile)) if load_profile[i] > 0.7]
    print(f"负载高峰时段 (>0.7): {peak_load_hours}")

    print("\n" + "="*50)
    print("保存配置文件和可视化")
    print("="*50)

    # 收集所有4个用户的配置文件数据
    all_customers_data = {}
    load_profiles_array = []
    generation_profiles_array = []
    customer_ids_list = []

    for customer_id in index:
        if f'customer_{customer_id}_load_profile' in globals():
            load_prof = globals()[f'customer_{customer_id}_load_profile']
            gen_prof = globals()[f'customer_{customer_id}_generation_profile']
            
            all_customers_data[customer_id] = {
                'load_profile': load_prof,
                'generation_profile': gen_prof
            }
            
            load_profiles_array.append(load_prof)
            generation_profiles_array.append(gen_prof)
            customer_ids_list.append(customer_id)

    # 转换为numpy数组
    load_profiles_array = np.array(load_profiles_array)
    generation_profiles_array = np.array(generation_profiles_array)

    # 保存为npy文件
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    np.save(load_npy_path, load_profiles_array)
    np.save(gen_npy_path, generation_profiles_array)
    np.save(ids_npy_path, np.array(customer_ids_list))

    print(f"✅ 已保存配置文件:")
    print(f"   - 负载配置文件: {load_npy_path} (形状: {load_profiles_array.shape})")
    print(f"   - 发电配置文件: {gen_npy_path} (形状: {generation_profiles_array.shape})")
    print(f"   - 客户ID: {ids_npy_path} (客户: {customer_ids_list})")

# 可视化4个用户的配置文件
import matplotlib.pyplot as plt

# ============================
# 图形风格设置（符合IEEE常用规范）
# ============================
# 字体：IEEE 推荐使用 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
# 线宽、字号等基础参数
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['axes.unicode_minus'] = False

# 创建图形（接近论文单栏宽度）
fig, axes = plt.subplots(2, 2, figsize=(7, 5))
# fig.suptitle('24-hour Normalized Load and Generation Profiles of 4 Grids',
            #  fontsize=10, fontweight='bold')

# 时间轴标签
hours = list(range(24))
hour_labels_plot = [f'{h:02d}:00' for h in hours]

for i, (customer_id, ax) in enumerate(zip(customer_ids_list, axes.flat)):
    load_data = load_profiles_array[i]
    generation_data = generation_profiles_array[i]
    
    # 上半部分：负载（正值），使用颜色显示
    ax.bar(
        hours,
        load_data,
        width=0.8,
        facecolor='#1f77b4',  # 蓝色
        edgecolor='black',
        label='Load',
        linewidth=0.8,
        alpha=0.9,
    )
    
    # 下半部分：发电（负值显示），使用另一种颜色显示
    ax.bar(
        hours,
        -generation_data,
        width=0.8,
        facecolor='#ff7f0e',  # 橙色
        edgecolor='black',
        label='Generation',
        linewidth=0.8,
        alpha=0.9,
    )
    
    # 子图标题改为 Grid 1, Grid 2, Grid 3, Grid 4，不再显示具体 ID
    ax.set_title(f'Grid {i + 1}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Time')
    
    # IEEE 一般不建议过重的网格线，这里用细虚线弱化处理
    ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.4)
    
    # 轴刻度和边框
    ax.tick_params(direction='in', length=3, width=0.8)
    
    # 设置横轴标签
    ax.set_xticks([0, 4, 8, 12, 16, 20, 23])
    ax.set_xticklabels(['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '23:00'])
    
    # 添加y=0的参考线
    ax.axhline(y=0, color='black', linewidth=0.8)
    
    # 设置y轴范围，给一点上下边距
    max_val = max(max(load_data), max(generation_data))
    ax.set_ylim(-max_val * 1.1, max_val * 1.1)
    
    # 图例放在右下方，去掉边框
    leg = ax.legend(frameon=False, loc='lower right')

# 布局紧凑，并预留标题空间
plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
# 提高分辨率，适用于论文插图
plt.savefig('./Dataset/customer_profiles_visualization.png', dpi=1200, bbox_inches='tight')
plt.show()

print(f"\n✅ 可视化图表已保存: ./Dataset/customer_profiles_visualization.png")
