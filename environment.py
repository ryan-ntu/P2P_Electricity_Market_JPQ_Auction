import gymnasium as gym
import numpy as np
from scipy.sparse import dok_matrix

class MultiMicroGridEnv(gym.Env):
    """
    Two-phase multi-agent microgrid environment:
    1) Day-ahead dispatch -> get_state
    2) Bidding transaction -> bidding_step
    CTDE-compatible: returns dicts for obs, reward, dones.
    """
    metadata = {'render.modes': []}
    def __init__(self, env_config, agents):
        super().__init__()
        self.config = env_config
        self.agents = agents
        self.n_agents = len(agents)
        self.max_steps = env_config['max_steps']
        self.market_mechanism = env_config['market_mechanism']
        self.current_step = 0
        self.done = False
        self.global_net_power = 0

        # Price parameters
        self.prices = {
            'common_price': env_config['common_price'],
            'emergency_price': env_config['emergency_price'],
            'feed_in_price': env_config['feed_in_price']
        }

        # Initialize hourly emergency price schedule (24h)
        self.emergency_price_hourly = self._init_emergency_price_schedule(
            env_config['emergency_price'], self.prices['common_price']
        )

        # Generation disruption configuration for robustness testing
        self.disruption_config = env_config.get('generation_disruption', {
            'enabled': True,  # Enable/disable disruption simulation
            'probability': 0.01,  # 1% chance of disruption per agent per hour
            'min_severity': 0.2,  # Minimum 20% reduction
            'max_severity': 0.8,  # Maximum 80% reduction
            'disruption_types': ['sudden_drop', 'gradual_decline', 'complete_failure'],
            'type_weights': [0.85, 0.1, 0.05],  # Probability weights for each type
            'persistence_probability': 0.1,  # 30% chance of persistent effects
            'max_persistence_hours': 4,  # Maximum hours for persistent effects
            'verbose_logging': False  # Enable detailed logging
        })

        # Observation and action spaces
        self.observation_space = gym.spaces.Dict({
            f'agent_{i}': gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(agent.observation_space,), 
                dtype=np.float32
            ) for i, agent in enumerate(self.agents)
        })
        self.action_space = gym.spaces.Dict({
            f'agent_{i}': gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(agent.action_space,),  
                dtype=np.float32
            ) for i, agent in enumerate(self.agents)
        })


    def _get_state(self):
        time_step = self.current_step % 24  # 当前小时 0-23
        state = {}

        def get_window(arr, t):
            """
            取 (t-1, t, t+1, t+2, t+3, t+4, t+5)，边界处用 clamp 处理（重复端点）
            arr: shape (24,)
            返回 shape (7,)
            """
            idxs = idxs = [ (t - 1) % 24, t % 24, (t + 1) % 24, (t + 2) % 24, (t + 3) % 24, (t + 4) % 24, (t + 5) % 24 ]
            return arr[idxs]

        for idx, agent in enumerate(self.agents):
            noise = np.zeros((2, 24), dtype=np.float32)  # 只给 demand 和 res_generation 加未来噪声

            if time_step < 23:
                future_len = 23 - time_step
                sampled = np.clip(
                    np.random.normal(0, np.linspace(0.01, 0.2, future_len), size=(2, future_len)),
                    a_min=-0.5,
                    a_max=0.5,
                )
                noise[:, time_step + 1 :] = sampled  # 填充未来噪声

            # 原始 full-length time-series
            obsever_demand_full = np.clip(agent.current_load_file + noise[0, :], 0, agent.demand_param)  # (24,)
            obsever_res_generation_full = np.clip(agent.current_generation_file + noise[1, :], 0, agent.generation_param)  # (24,)

            common_purchase_full = agent.common_purchase  # 应该是 shape (24,)

            # scalar features 
            scalar_features = np.array([
                self.global_net_power,
                agent.current_net_power,
                agent.storage / agent.parameter_battery,
            ], dtype=np.float32)  # (3,)

            # 时序 window: emergency_price, demand, res_generation, common_purchase

            emergency_price_w = get_window(self.emergency_price_hourly, time_step)              # (7,)
            demand_w = get_window(obsever_demand_full, time_step)              # (7,)
            resgen_w = get_window(obsever_res_generation_full, time_step)      # (7,)
            common_purchase_w = get_window(common_purchase_full, time_step)    # (7,)


            # stack 成 (7,3)，再 flatten 为 21
            window_7x4 = np.stack([emergency_price_w, demand_w, resgen_w, common_purchase_w], axis=1)  # (7,4)
            window_flat = window_7x4.reshape(-1)  # (28,)

            # time-of-day 周期编码 sin/cos
            hour = float(time_step)  # 0..23
            theta = 2 * np.pi * (hour / 24.0)
            sin_t = np.sin(theta).astype(np.float32)
            cos_t = np.cos(theta).astype(np.float32)
            time_feat = np.array([sin_t, cos_t], dtype=np.float32)  # (2,)

            # 组成最终 state vector：3 scalar + 28 window + 2 time = 33
            state_vector = np.concatenate([
                scalar_features,   # 3
                window_flat,       # 28
                time_feat,         # 2
            ], axis=0).astype(np.float32)  # (33,)

            state[f'agent_{idx}'] = state_vector

        return state

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.global_net_power = 0
        self.done = False
        for agent in self.agents:
            agent.reset()
    



    def day_ahead_dispatch(self):
        """
        Executes the day-ahead dispatch for all agents.
        Agents: Demand prediction and Res prediction --> Net power calculation
        and 
        Executes the calculation of the intra-day gap for all agents.
        Agents: Real demand and Real Res generation plus the purchase amount --> Net power calculation
        """ 
        load_error = np.random.normal(loc=0.0, scale=np.linspace(0.01, 0.3, 24))
        weather_error = np.random.normal(0.0, scale=np.linspace(0.01, 0.2, 24))

        for agent in self.agents:
            agent.reset(semi=True)
            agent.current_load_file = np.clip(agent.load_file + np.random.normal(0, 0.1, 24), 0, 1)
            agent.current_generation_file = np.clip(agent.generation_file + np.random.normal(0, 0.1, 24), 0, 1)

            agent.demand_prediction(load_error)
            agent.res_prediction(weather_error)
            agent.get_plan(self.prices)

    
    def _apply_generation_disruption(self):
        """
        Applies random generation disruption to simulate equipment failures,
        weather changes, or other unexpected events that affect power generation.
        
        This method modifies the current generation values of agents to test
        the robustness of the P2P bidding system.
        """
        # Check if disruption simulation is enabled
        if not self.disruption_config.get('enabled', False):
            return
            
        current_hour = self.current_step % 24
        
        for agent in self.agents:
            # Check if disruption occurs
            if np.random.random() < self.disruption_config['probability']:
                # Select disruption type
                disruption_type = np.random.choice(
                    self.disruption_config['disruption_types'], 
                    p=self.disruption_config['type_weights']
                )
                
                # Generate disruption severity
                disruption_severity = np.random.uniform(
                    self.disruption_config['min_severity'], 
                    self.disruption_config['max_severity']
                )
                
                # Apply disruption based on type
                original_generation = agent.res_generation[current_hour]
                
                if disruption_type == 'sudden_drop':
                    # Sudden drop: immediate significant reduction
                    disrupted_generation = original_generation * (1 - disruption_severity)
                    
                elif disruption_type == 'gradual_decline':
                    # Gradual decline: smaller reduction that might persist
                    disruption_severity *= 0.5  # Less severe for gradual decline
                    disrupted_generation = original_generation * (1 - disruption_severity)
                    
                elif disruption_type == 'complete_failure':
                    # Complete failure: near-zero generation
                    disrupted_generation = original_generation * 0.05  # 5% of original
                    
                # Ensure generation doesn't go negative
                disrupted_generation = max(0, disrupted_generation)
                
                # Update the generation for current hour
                agent.res_generation[current_hour] = disrupted_generation
                
                # Log the disruption if verbose logging is enabled
                # if self.disruption_config.get('verbose_logging', False):
                #     reduction_percentage = ((original_generation - disrupted_generation) / original_generation) * 100
                #     print(f"Generation disruption ({disruption_type}) applied to agent {agent.id} at hour {current_hour}: "
                #           f"Original: {original_generation:.2f}, Disrupted: {disrupted_generation:.2f} "
                #           f"(Reduction: {reduction_percentage:.1f}%)")
                
                # Optionally, apply persistent effects to future hours
                if (disruption_type == 'gradual_decline' and 
                    np.random.random() < self.disruption_config.get('persistence_probability', 0.3)):
                    # Apply persistent effect for next few hours
                    persistence_hours = np.random.randint(1, self.disruption_config.get('max_persistence_hours', 3) + 1)
                    for future_hour in range(current_hour + 1, min(current_hour + 1 + persistence_hours, 24)):
                        future_original = agent.res_generation[future_hour]
                        future_disrupted = future_original * (1 - disruption_severity * 0.3)  # Reduced effect
                        agent.res_generation[future_hour] = max(0, future_disrupted)
                        
                        # if self.disruption_config.get('verbose_logging', False):
                        #     print(f"  Persistent effect: hour {future_hour}, generation reduced to {future_disrupted:.2f}")

    def p2p_bidding_preparation(self):
        """
        Executes the p2p bidding preparation for all agents.
        Includes generation disruption simulation for robustness testing.
        """
        net_power_list = []
        
        # Apply generation disruption for robustness testing
        self._apply_generation_disruption()
        
        for agent in self.agents:
            agent.reset(semi=True)
            
            agent.net_calculation(time_step=self.current_step % 24)
            net_power_list.append(agent.current_net_power-agent.storage)

        self.global_net_power = np.sum(np.array(net_power_list), axis=0)

        return self._get_state()

    def bidding_step(self, qty_list):
        """
        Executes the bidding step for all agents.
        """
        self.bidding_transaction(qty_list)

        # update the state and reward
        self.reward = {f'agent_{i}': agent.first_reward for i, agent in enumerate(self.agents)}

        self.current_step += 1
        terminated = {f'agent_{i}': False for i in range(self.n_agents)}
        truncated = {f'agent_{i}': (self.current_step >= self.max_steps) for i in range(self.n_agents)}

        return self._get_state(),  self.reward, terminated, truncated, {}
    
    def bidding_transaction(self, qty_list):
        bidding_price_list = []
        for agent in self.agents:
            bidding_price_list.append(agent.price)
            agent.community_cost = 0    # set the community cost to 0 in case of calculation error
        
        if self.config['market_mechanism'] == 'vcg':
            alloc_qty, payments = self.proposed_allocate_power(qty_list, bidding_price_list, mechanism=self.config['market_mechanism'])
            for i in range(self.n_agents):
                received = alloc_qty[i, :].sum()
                self.agents[i].settle_trade(bought=received, total_price=0)
                sold = alloc_qty[:, i].sum()
                self.agents[i].settle_trade(sold=sold, total_price=payments[i])
        else:
            alloc_qty, alloc_price = self.proposed_allocate_power(qty_list, bidding_price_list, mechanism=self.config['market_mechanism'])

            for i in range(self.n_agents):
                received = alloc_qty[i, :].sum()
                transaction_price = -1*(alloc_price[i, :].multiply(alloc_qty[i, :])).sum()
                self.agents[i].settle_trade(bought=received, total_price=transaction_price)

            for j in range(self.n_agents):
                sold = alloc_qty[:, j].sum()
                transaction_price = (alloc_price[:, j].multiply(alloc_qty[:, j])).sum()
                self.agents[j].settle_trade(sold=sold, total_price=transaction_price)

        # Inject time-varying emergency price for this hour to agents
        current_prices = dict(self.prices)
        current_prices['emergency_price'] = self.get_emergency_price(self.current_step % 24)
        for agent in self.agents:
            agent.update_after_trade(unit_price=current_prices, time_step=self.current_step % 24)
            # should record the data needed of agents for learning

    def proposed_allocate_power(self, power_list, bidding_price_list, mechanism='simple'):
        """
        Allocates the power to the agents based on the bidding price.
        Args:
            power_list: List of net power for each agent
            bidding_price_list: List of bidding prices for each agent
            mechanism: 'simple' for simple matching or 'vcg' for VCG mechanism
        """
        if mechanism == 'simple':
            return self._simple_matching(power_list, bidding_price_list)
        elif mechanism == 'mrda':
            return self._mrda_mechanism(power_list, bidding_price_list)
        elif mechanism == 'msmrda':
            return self._msmrda_mechanism(power_list, bidding_price_list)
        elif mechanism == 'vcg':
            return self._vcg_mechanism(power_list, bidding_price_list)
        elif mechanism == 'vda':
            return self._vickrey_double_auction(power_list, bidding_price_list)
        else:
            raise ValueError("Unsupported mechanism")

    def _simple_matching(self, power_list, bidding_price_list):
        """
        Original proposed matching mechanism
        """
        net_power_sheet = np.array(power_list)
        bidding_price_sheet = np.array(bidding_price_list)
        N = len(net_power_sheet)
        if -30< np.sum(self.global_net_power) < -20:
            total = 0
        elif np.sum(self.global_net_power) <= -30:
            total = -1
        else:
            total = 1

        buyers = []
        sellers = []
        """construct the buyers and sellers list"""
        for i in range(N):
            if net_power_sheet[i] > 0:
                buyers.append({'id': i,
                            'rem': net_power_sheet[i],
                            'price': bidding_price_sheet[i],
                            'key1': bidding_price_sheet[i] * net_power_sheet[i],
                            'key2': bidding_price_sheet[i]})
                
            elif net_power_sheet[i] < 0:
                sellers.append({'id': i,
                                'rem': -net_power_sheet[i],
                                'price': bidding_price_sheet[i],
                                'key1': bidding_price_sheet[i],
                                'key2': (self.get_emergency_price(self.current_step % 24) - bidding_price_sheet[i]) * (-net_power_sheet[i])})
                
        if total < 0:
            buyers.sort(key=lambda x: x['key1'], reverse=True)
            sellers.sort(key=lambda x: x['key1'])
        elif total==0:
            buyers.sort(key=lambda x: x['key2'], reverse=True)
            sellers.sort(key=lambda x: x['key1'])
        else:
            buyers.sort(key=lambda x: x['key2'], reverse=True)
            sellers.sort(key=lambda x: x['key2'], reverse=True)
        
        alloc_qty = dok_matrix((N, N), dtype=float)
        alloc_price = dok_matrix((N, N), dtype=float)

        bi, si = 0, 0
        bi_start, si_start = 0, 0

        while bi < len(buyers) and si < len(sellers):
            b = buyers[bi]
            s = sellers[si]
            if b['price'] < s['price']:
                if total < 0:
                    bi += 1
                    bi_start += 1
                    continue
                elif total >0:
                    si += 1
                    si_start += 1
                    continue
                else:
                    break
            else:
                traded_qty = min(b['rem'], s['rem'])
                alloc_qty[b['id'], s['id']] = traded_qty
                alloc_price[b['id'], s['id']] = (b['price']+s['price'])/2.0
                b['rem'] -= traded_qty
                s['rem'] -= traded_qty
                if np.isclose(b['rem'], 0):
                    bi_start += 1
                if np.isclose(s['rem'], 0):
                    si_start += 1

            if bi_start == len(buyers) or si_start == len(sellers):
                break

            bi += 1
            si += 1
            
            if bi >= len(buyers):
                bi = bi_start
            if si >= len(sellers):
                si = si_start

        return alloc_qty.tocsr(), alloc_price.tocsr()

    def _init_emergency_price_schedule(self, emergency_price_config, common_price):
        """
        Initialize a 24-hour emergency price vector.
        - If a 24-length list/array is provided, use it (clamped with a floor over common price).
        - If a scalar is provided, generate a reasonable TOU curve around it.
        """
        # If already an hourly schedule
        if isinstance(emergency_price_config, (list, tuple, np.ndarray)):
            arr = np.array(emergency_price_config, dtype=np.float32).flatten()
            if arr.size != 24:
                raise ValueError("emergency_price must be a float or a 24-length array")
            # Ensure emergency price stays above common grid price by a small margin
            floor = float(common_price) * 1.5
            arr = np.maximum(arr, floor)
            return arr

        # Scalar -> build a reasonable TOU curve
        base = float(emergency_price_config)
        # Multipliers: off-peak (0-6), morning peak (7-10), midday (11-16), evening peak (17-21), late (22-23)
        multipliers = np.array([
            0.80, 0.78, 0.76, 0.75, 0.75, 0.78, 0.85,  # 00-06
            1.00, 1.05, 1.1, 1.10,                    # 07-11
            1.20, 1.30, 1.35, 1.4, 1.45, 1.45,        # 12-17
            1.35, 1.30, 1.25, 1.20, 1.15,              # 18-22
            0.90, 0.85                                  # 22-23
        ], dtype=np.float32)
        if multipliers.size != 24:
            raise RuntimeError("Emergency price multipliers must have length 24")
        arr = base * multipliers
        # Keep emergency price strictly above common price (as a penalty price)
        floor = float(common_price) * 1.20
        arr = np.maximum(arr, floor)
        return arr

    def get_emergency_price(self, time_step: int) -> float:
        """Return the emergency price for a given hour (0-23)."""
        return float(self.emergency_price_hourly[int(time_step) % 24])

    @staticmethod
    def _mrda_mechanism(power_list, bidding_price_list):
        """
        MRDA mechanism implementation
        """
        net_power_sheet = np.array(power_list)
        bidding_price_sheet = np.array(bidding_price_list)
        N = len(net_power_sheet)
        buyers = []
        sellers = []
        """construct the buyers and sellers list"""
        for i in range(N):
            if net_power_sheet[i] > 0:
                buyers.append({'id': i,
                            'rem': net_power_sheet[i],
                            'price': bidding_price_sheet[i]})
                
            elif net_power_sheet[i] < 0:
                sellers.append({'id': i,
                                'rem': -net_power_sheet[i],
                                'price': bidding_price_sheet[i]})
        
        buyers.sort(key=lambda x: x['price'], reverse=True)
        sellers.sort(key=lambda x: x['price'])
        
        alloc_qty = dok_matrix((N, N), dtype=float)
        alloc_price = dok_matrix((N, N), dtype=float)
        bi, si = 0, 0
        
        while bi < len(buyers) and si < len(sellers):
            b = buyers[bi]
            s = sellers[si]
            if b['price'] < s['price']:
                break
            else:
                traded_qty = min(b['rem'], s['rem'])
                alloc_qty[b['id'], s['id']] = traded_qty
                alloc_price[b['id'], s['id']] = (b['price']+s['price'])/2.0

                b['rem'] -= traded_qty
                s['rem'] -= traded_qty

                if np.isclose(b['rem'], 0):
                    bi += 1 
                if np.isclose(s['rem'], 0):
                    si += 1

        return alloc_qty.tocsr(), alloc_price.tocsr()

    @staticmethod
    def _msmrda_mechanism(power_list, bidding_price_list):
        """
        Multi-step MRDA mechanism implementation
        """
        net_power_sheet = np.array(power_list)
        bidding_price_sheet = np.array(bidding_price_list)
        N = len(net_power_sheet)

        buyers = []
        sellers = []
        """construct the buyers and sellers list"""
        for i in range(N):
            if net_power_sheet[i] > 0:
                buyers.append({'id': i,
                            'rem': net_power_sheet[i],
                            'price': bidding_price_sheet[i]})
                
            elif net_power_sheet[i] < 0:
                sellers.append({'id': i,
                                'rem': -net_power_sheet[i],
                                'price': bidding_price_sheet[i]})
        
        buyers.sort(key=lambda x: x['price'], reverse=True)
        sellers.sort(key=lambda x: x['price'])
        
        alloc_qty = dok_matrix((N, N), dtype=float)
        alloc_price = dok_matrix((N, N), dtype=float)
        bi, si = 0, 0
        bi_start, si_start = 0, 0

        while bi < len(buyers) and si < len(sellers):
            b = buyers[bi]
            s = sellers[si]
            if b['price'] < s['price']:
                if bi_start == bi and si_start == si:
                    break
                else:
                    pass
            else:
                traded_qty = min(b['rem'], s['rem'])
                alloc_qty[b['id'], s['id']] = traded_qty
                alloc_price[b['id'], s['id']] = (b['price']+s['price'])/2.0
                b['rem'] -= traded_qty
                s['rem'] -= traded_qty
                if np.isclose(b['rem'], 0):
                    bi_start += 1
                if np.isclose(s['rem'], 0):
                    si_start += 1
            bi += 1
            si += 1
            
            if bi >= len(buyers):
                bi = bi_start
            if si >= len(sellers):
                si = si_start

        return alloc_qty.tocsr(), alloc_price.tocsr()

    @staticmethod
    def _vcg_mechanism(power_list, bidding_price_list):
        """
        VCG mechanism implementation
        """
        net_power = np.array(power_list, dtype=float)
        prices = np.array(bidding_price_list, dtype=float)
        N = len(net_power)

        # build buyers and sellers lists
        buyers = [{'id': i, 'rem': net_power[i], 'price': prices[i]} for i in range(N) if net_power[i] > 0]
        sellers = [{'id': i, 'rem': -net_power[i], 'price': prices[i]} for i in range(N) if net_power[i] < 0]
        buyers.sort(key=lambda x: x['price'], reverse=True)
        sellers.sort(key=lambda x: x['price'])

        # initial allocation via greedy matching
        alloc_qty = dok_matrix((N, N), dtype=float)
        alloc_pairs = []  # keep track of matches
        bi, si = 0, 0
        while bi < len(buyers) and si < len(sellers):
            b, s = buyers[bi], sellers[si]
            qty = min(b['rem'], s['rem'])
            if qty <= 0:
                break
            alloc_qty[b['id'], s['id']] = qty
            alloc_pairs.append((b['id'], s['id'], qty))
            b['rem'] -= qty
            s['rem'] -= qty
            if b['rem'] == 0:
                bi += 1
            if s['rem'] == 0:
                si += 1

        # compute overall welfare
        def welfare(matrix):
            W = 0.0
            for (i, j), q in matrix.items():
                W += (prices[i] - prices[j]) * q
            return W

        W_star = welfare(alloc_qty)

        # compute traded quantities per agent
        traded_qty = np.zeros(N, dtype=float)
        for i, j, q in alloc_pairs:
            traded_qty[i] += q
            traded_qty[j] += q

        # compute VCG payments per agent
        payments = np.zeros(N, dtype=float)
        for k in range(N):
            # remove agent k
            power2 = net_power.copy()
            price2 = prices.copy()
            power2[k] = 0.0
            price2[k] = 0.0
            # rebuild lists and reallocate
            b2 = [{'id': i, 'rem': power2[i], 'price': price2[i]} for i in range(N) if power2[i] > 0]
            s2 = [{'id': i, 'rem': -power2[i], 'price': price2[i]} for i in range(N) if power2[i] < 0]
            b2.sort(key=lambda x: x['price'], reverse=True)
            s2.sort(key=lambda x: x['price'])
            alloc2 = dok_matrix((N, N), dtype=float)
            bi2, si2 = 0, 0
            while bi2 < len(b2) and si2 < len(s2):
                bb, ss = b2[bi2], s2[si2]
                q2 = min(bb['rem'], ss['rem'])
                if q2 <= 0:
                    break
                alloc2[bb['id'], ss['id']] = q2
                bb['rem'] -= q2
                ss['rem'] -= q2
                if bb['rem'] == 0:
                    bi2 += 1
                if ss['rem'] == 0:
                    si2 += 1
            W_minus = welfare(alloc2)
            # agent's self value
            if net_power[k] > 0:
                v_k = prices[k] * traded_qty[k]
            else:
                v_k = -prices[k] * traded_qty[k]
            payments[k] = W_minus - (W_star - v_k)

        return alloc_qty.tocsr(), payments
    
    @staticmethod
    def _vickrey_double_auction(power_list, price_list):
        """
        双边 Vickrey 拍卖（第二价格双侧拍卖）实现：
        - 买方按出价降序排列，卖方按出价（要价）升序排列
        - 找到最大 k 使得 b_k >= s_k
        - 成交量为 k，对前 k 个买家/卖家分别匹配
        - 每个买家支付第 k+1 买价（b_{k+1}），每个卖家获得第 k+1 卖价（s_{k+1}）
        """

        bids = np.array(price_list)
        qtys = np.array(power_list)
        N = len(bids)

        # 构造买家/卖家列表
        buyers = [(i, qtys[i], bids[i]) for i in range(N) if qtys[i] > 0]
        sellers = [(i, -qtys[i], bids[i]) for i in range(N) if qtys[i] < 0]
        if not buyers or not sellers:
            # 无匹配
            return dok_matrix((N, N)), dok_matrix((N, N))

        # 排序
        buyers.sort(key=lambda x: x[2], reverse=True)    # 按价格从高到低
        sellers.sort(key=lambda x: x[2])                 # 按价格从低到高

        # 找到 k
        k = 0
        while k < min(len(buyers), len(sellers)) and buyers[k][2] >= sellers[k][2]:
            k += 1
        k = k  # 成交对数

        if k == 0:
            # 无成交
            return dok_matrix((N, N)), dok_matrix((N, N))

        # 第 k+1 价格
        next_buy_price = buyers[k][2] if k < len(buyers) else buyers[-1][2]
        next_sell_price = sellers[k][2] if k < len(sellers) else sellers[-1][2]

        # 分配矩阵
        alloc_qty = dok_matrix((N, N), dtype=float)
        alloc_price = dok_matrix((N, N), dtype=float)

        # 对前 k 对进行匹配
        for idx in range(k):
            bi, bqty, _ = buyers[idx]
            si, sqty, _ = sellers[idx]
            traded = min(bqty, sqty)
            alloc_qty[bi, si] = traded
            # 买家支付 next_sell_price，卖家收取 next_buy_price
            # 双边 Vickrey：通常买家支付下一卖价，卖家收取下一买价
            alloc_price[bi, si] = 0.5 * (next_buy_price + next_sell_price)

        return alloc_qty.tocsr(), alloc_price.tocsr()



    def action_to_bid(self, action, is_continuous=True):
        """
        Converts the action to bidding price and quantity.
        
        Args:
            action: Dict containing actions for each agent
            is_continuous: Whether using continuous action space
            
        Returns:
            price_list: List of bidding prices for each agent
            qty_list: List of bidding quantities for each agent
            battery_control_param: List of battery control parameters for each agent
        """
        price_list = []
        qty_list = []
        
        if is_continuous:
            for aid in self.agents:
                agent_action_raw = action[f'agent_{aid.id}']
                
                # Validate action shape
                if agent_action_raw.ndim != 1 or agent_action_raw.shape[0] != 3:
                    raise ValueError(f"Agent {aid.id} action should be 1D array with 2 elements")
                
                # Normalize action to [-1, 1] for direction and [0, 1] for price ratio
                direction = 2 * agent_action_raw[0] - 1  # [-1, 1]: negative=sell, positive=buy
                price_ratio = np.clip(agent_action_raw[1], 0.0, 1.0)  # [0, 1]
                aid.battery_control_param = np.clip(agent_action_raw[2], 0.0, 1.0)  # [0, 1]
                # Calculate bidding price
                current_hour = self.current_step % 24
                floor_price = float(self.prices['feed_in_price'])
                ceil_price = float(self.get_emergency_price(current_hour))
                
                # Handle edge case where emergency price is not higher than feed-in price
                if ceil_price <= floor_price:
                    ceil_price = floor_price * 5  # More reasonable multiplier
                
                aid.price = np.round(floor_price + (ceil_price - floor_price) * price_ratio, 2)
                price_list.append(aid.price)
                
                # Calculate bidding quantity based on direction and constraints
                bid_qty = self._calculate_bid_quantity(aid, direction)
                qty_list.append(bid_qty)
        else:
            # Discrete action space handling
            action_dict = {aid: int(action[f'agent_{aid.id}']) for aid in self.agents}
            # TODO: Implement discrete action logic if needed
            raise NotImplementedError("Discrete action space not implemented")

        
        return price_list, qty_list
    
    def _calculate_bid_quantity(self, agent, direction):
        """
        Calculate bidding quantity based on agent state and direction.
        
        Args:
            agent: The agent object
            direction: Direction of trade (-1 for sell, 1 for buy)
            
        Returns:
            float: Bidding quantity (positive for buy, negative for sell)
        """
        if direction >= 0:  # Buy direction
            return self._calculate_buy_quantity(agent, direction)
        else:  # Sell direction
            return self._calculate_sell_quantity(agent, abs(direction))
    
    def _calculate_buy_quantity(self, agent, direction_ratio):
        """
        Calculate buy quantity considering battery constraints.
        
        Args:
            agent: The agent object
            direction_ratio: Ratio of maximum buy quantity to bid [0, 1]
            
        Returns:
            float: Buy quantity (positive)
        """
        # Available charging capacity

        max_buy_qty = max(0, agent.charge_lim + agent.current_net_power)
        
        return max_buy_qty * direction_ratio
    
    def _calculate_sell_quantity(self, agent, direction_ratio):
        """
        Calculate sell quantity considering battery and generation constraints.
        
        Args:
            agent: The agent object
            direction_ratio: Ratio of maximum sell quantity to bid [0, 1]
            
        Returns:
            float: Sell quantity (negative)
        """
       # Agent has excess power, can sell
        available_sell_capacity = max( 0,  agent.discharge_lim - agent.current_net_power )
            
        return -available_sell_capacity * direction_ratio  

      








