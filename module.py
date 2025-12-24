import numpy as np

class micro_grid_agent(object):
    def __init__(self, parameter_panel):
        # System parameters
        self.generation_param = parameter_panel["generation_param"]
        self.demand_param = parameter_panel["demand_param"]
        self.parameter_battery = parameter_panel["battery_lim"]
        self.charge_lim = parameter_panel["charge_lim"]
        self.discharge_lim = parameter_panel["discharge_lim"]
        self.id = parameter_panel["id"]
        self.action_space = parameter_panel["action_space"]
        self.load_file = np.load(parameter_panel["load_file"])[self.id]
        self.current_load_file = None
        self.generation_file = np.load(parameter_panel["generation_file"])[self.id]
        self.current_generation_file = None
        
        # Current state variables
        self.storage = parameter_panel["initial_storage"]
        self.demand = 0
        self.res_generation = 0
        self.pre_demand = 0
        self.pre_res_generation = 0
        self.common_purchase = None
        self.battery_charge = 0
        self.battery_discharge = 0
        
        self.sold_power = 0
        self.bought_power = 0
        self.current_net_power = 0
        self.social_welfare = 0
        self.emergency_purchase = 0
        self.feed_in_power = 0
        self.price = np.random.uniform(1, 10)
        self.battery_control_param = 1.0

        # Price related attributes
        self.cost = 0
        # Cost related to transactions with the grid
        self.grid_cost = 0 
        # Cost related to bidding with local community
        self.community_cost = 0

        # RL related attributes
        self.first_reward = 0
        self.second_reward = 0
        # action: bidding price
        
        # observation: storage, demand, res_generation, common_purchase plus weather_factor*2 and total_net_power
        self.observation_space = 33
        self.reward = 0   # it is the reward of bidding  model, should be the cost of the second stage bidding, including community cost and emergency cost.
        
        # Decision Model Day-ahead

        
        self.day_ahead_planer = None
    
    def reset(self, semi=False):
        """
        Resets the agent's state
        """
        if not semi:
            self.storage = np.random.uniform(0, self.parameter_battery)
            self.demand = 0
            self.res_generation = 0
            self.pre_demand = 0
            self.pre_res_generation = 0

        self.emergency_purchase = 0
        self.feed_in_power = 0
        self.sold_power = 0
        self.bought_power = 0
        self.battery_discharge = 0
        self.battery_charge = 0
        self.battery_control_param = 1.0
        self.current_net_power = 0

        self.cost = 0
        self.grid_cost = 0
        self.community_cost = 0


    def demand_prediction(self, load_factor: np.ndarray):
        """
        预测明天一天 24 小时的负载。

        参数说明
        ----------
        load_factor : np.ndarray
            长度为 24 的随机误差系数向量 ，元素范围被环境裁剪在 [-0.8, 0.8]。
        """

        self.pre_demand = self.demand_param * np.clip(self.current_load_file + load_factor, 0, 1)
        self.demand = self.demand_param * self.current_load_file


    def res_prediction(self, weather_factor: np.ndarray):
        """
        预测明天一天内 24 小时的可再生能源发电量。

        参数说明
        ----------
        weather_factor : np.ndarray
            长度为 24 的随机误差系数向量 ，元素范围被环境裁剪在 [-0.8, 0.8]。
        """

        self.pre_res_generation = self.generation_param * np.clip(self.current_generation_file + weather_factor, 0, 1)
        self.res_generation = self.generation_param * self.current_generation_file

    

    def net_calculation(self, time_step, update=False):
        self.current_net_power = self.demand[time_step] - self.res_generation[time_step] - self.common_purchase[time_step]
        
        if update:
            self.current_net_power += self.sold_power - self.bought_power

            if self.current_net_power > 0:
                if self.battery_discharge < self.discharge_lim:
                    single_discharge = min(self.current_net_power, self.discharge_lim-self.battery_discharge, self.storage)
                    self.storage -= single_discharge
                    self.battery_discharge += single_discharge
                    self.current_net_power -= single_discharge
            else:
                if self.storage >= self.parameter_battery * self.battery_control_param:
                    discharge = min(self.storage - self.parameter_battery * self.battery_control_param, self.discharge_lim-self.battery_discharge)
                    self.storage -= discharge
                    self.battery_discharge += discharge
                    self.current_net_power -= discharge
                elif self.battery_charge < self.charge_lim:
                    single_charge = min(np.abs(self.current_net_power), self.charge_lim-self.battery_charge, self.battery_control_param*self.parameter_battery-self.storage)
                    self.storage += single_charge
                    self.battery_charge += single_charge
                    self.current_net_power += single_charge

    def get_plan(self, price):
        """
        Determines energy purchase plan using RL policy in day-ahead dispatch
        """
        
        self.common_purchase = 0.95 * (self.pre_demand - self.pre_res_generation)
        self.common_purchase[self.common_purchase < 0] = 0
        
        return self.common_purchase
        

    def cost_calculation(self, price, time_step):
        """
        Calculates the cost of the agent
        """
        first_grid_cost =  -1* self.common_purchase[time_step] * price['common_price']
        second_grid_cost = self.feed_in_power*price['feed_in_price']- self.emergency_purchase*price['emergency_price']

        return first_grid_cost, second_grid_cost

    
    def settle_trade(self, total_price, bought=0, sold=0):
        if bought > 0:
            self.bought_power += bought
        if sold > 0:
            self.sold_power += sold

        self.community_cost += total_price

    def update_after_trade(self, unit_price, time_step):
        self.net_calculation(time_step, update=True)
        if self.current_net_power >= 0:
            self.emergency_purchase = self.current_net_power
        else:
            self.feed_in_power = np.abs(self.current_net_power)
            
        first_grid_cost, second_grid_cost = self.cost_calculation(unit_price, time_step)

        self.first_reward = self.community_cost + second_grid_cost 
        # self.second_reward = self.social_welfare
        
