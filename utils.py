import csv
import os
import numpy as np

def init_training_csv(n_agents, log_dir='statistics', filename='training_log.csv', algorithm_name='MAPPO', market_mechanism='msmrda'):
    log_dir_fs = os.path.join(log_dir)
    os.makedirs(log_dir_fs, exist_ok=True)
    # 如果filename已经包含.csv扩展名，就不重复添加
    if filename.endswith('.csv'):
        csv_filename = f'{algorithm_name}_{market_mechanism}_{filename}'
    else:
        csv_filename = f'{algorithm_name}_{market_mechanism}_{filename}.csv'
    csv_path = os.path.join(log_dir_fs, csv_filename)
    file_exists = os.path.exists(csv_path)
    csv_f = open(csv_path, mode='a', newline='')
    csv_writer = csv.writer(csv_f)
    if not file_exists:
        header = ['episode']
        for i in range(n_agents):
            header += [
                f'agent_{i}_reward',
                f'agent_{i}_emergency_purchase',
                f'agent_{i}_feed_in_power',
                f'agent_{i}_storage_level',
                f'agent_{i}_cost',
            ]
        header += ['total_reward', 'total_emergency', 'total_feed_in', 'total_cost']
        csv_writer.writerow(header)
        csv_f.flush()
    return csv_writer, csv_f, csv_path


class OUNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.1, dt=1e-2, scale= None):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt 
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()
        self.scale = scale 

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) +  np.sqrt(self.dt) * self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        if self.scale is None:
            return self.state 
        else:
            return self.state * self.scale
