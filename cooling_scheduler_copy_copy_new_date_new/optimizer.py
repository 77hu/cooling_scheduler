# ============================================================================
# cooling_scheduler/optimizer.py
# 模块 5：动态规划（DP）优化器
#
# 职责：使用动态规划算法，求解未来 24h 冷却启停的最优计划，
#       使总电费最低，同时温度保持在安全范围内。
#
# 算法：
#   状态空间：四维（温度 × SoC × 上一小时冷却状态 p × BMS 充电锁存 l）
#   向后递推（Bellman 方程）：
#     V[h][t][s][p][l] = min over a∈{0,1}
#                        { cost(h,t,s,p,l,a) + V[h+1][t_next][s_next][a][l_next] }
#   前向追踪：从初始状态出发，按最优策略走一遍
#
# 计算复杂度：
#   H=24 × T=81 温度状态 × S=21 SoC 状态 × 2 前冷却状态 × 2 锁存 × 2 决策
#   = 326,592 次模型调用，预计耗时 2~5 秒
#
# 依赖：模块 1（config）、模块 3（thermal_model）
# 被依赖：模块 7（输出）、模块 8（可视化）、模块 9（对比）、模块 10（主程序）
# ============================================================================

import numpy as np
import time as time_module

from .config import SystemConfig
from .thermal_model import DetailedThermalModel, day_hour_of


class CoolingOptimizer:
    """
    动态规划冷却计划优化器（四维状态空间：温度 × SoC × 前冷却状态 × 充电锁存）。

    将 24 小时离散为 24 个阶段，每个阶段做决策：冷却开 or 关。
    状态 = (电解液温度, SoC, 上一小时冷却状态 p, BMS 充电锁存 l)，
    温度与 SoC 离散化为网格；p 使启停惩罚满足马尔可夫性，
    l（充电撞 SoC_max 后置位、窗口结束清除）使充放电判断与历史一致。

    通过向后递推求解 Bellman 方程，得到全局最优的冷却启停计划。
    SoC 动态变化影响充放电状态判断
    """

    def __init__(self, config: SystemConfig):
        """
        初始化优化器。

        参数:
            config: SystemConfig 实例，从中读取 optimizer 参数
        """
        opt = config.optimizer
        self.T_min = opt["T_min"]                 # [°C] 温度状态空间下限
        self.T_max = opt["T_max"]                 # [°C] 温度状态空间上限
        self.T_step = opt["T_step"]               # [°C] 温度离散步长
        self.T_target_low = opt["T_target_low"]   # [°C] 舒适下限
        self.T_target_high = opt["T_target_high"] # [°C] 舒适上限
        self.T_dyn_base = opt.get("T_dyn_base", 38.0)    # [°C] SoC=0 温度上限
        self.T_dyn_slope = opt.get("T_dyn_slope", 4.0)   # 随SoC每单位下降幅度
        self.horizon_h = opt["horizon_h"]         # [小时] 优化时域
        # self.run_penalty = opt["run_penalty"]     # [元/h] 运行时长惩罚系数
        self.startup_penalty=opt['startup_penalty']  # [元/次] 启停惩罚

        # SoC 状态空间参数
        self.SoC_min = 0.0                        # SoC 下限
        self.SoC_max = 1.0                        # SoC 上限
        self.SoC_step = 0.05                      # SoC 离散步长

        # 构建温度网格 [°C]：从 T_min 到 T_max，步长 T_step
        #   注：终点写成 T_max + T_step*0.5 是利用 arange 开区间特性，保证 T_max 这个端点被包含
        #   例：20 到 40、步长 0.25 → 20, 20.25, ..., 40.0 共 81 个格点
        self.T_grid = np.arange(self.T_min,
                                self.T_max + self.T_step * 0.5,
                                self.T_step)
        self.n_T = len(self.T_grid)      # 温度格点数 = 81

        # 构建 SoC 网格：0 到 1，步长 0.05 → 0, 0.05, ..., 1.0 共 21 个格点
        self.SoC_grid = np.arange(self.SoC_min,
                                  self.SoC_max + self.SoC_step * 0.5,
                                  self.SoC_step)
        self.n_SoC = len(self.SoC_grid)  # SoC 格点数 = 21

    def _nearest_T_idx(self, T):
        """连续温度 → 最近温度网格索引"""
        idx = int(round((T - self.T_min) / self.T_step))
        return int(np.clip(idx, 0, self.n_T - 1))

    def _nearest_SoC_idx(self, SoC):
        """连续 SoC → 最近 SoC 网格索引"""
        idx = int(round((SoC - self.SoC_min) / self.SoC_step))
        return int(np.clip(idx, 0, self.n_SoC - 1))

    def _terminal_penalty(self, T, SoC):
        """
        终值惩罚函数：引导 DP 避免极端温度和极端 SoC。

        设计思路：
          - 温度低于 T_target_low：轻微惩罚（避免过度冷却浪费电能）
          - 温度高于 T_target_high：较重惩罚（高温影响电池寿命和安全）
          - SoC 接近 0 或 1：额外惩罚（避免过充/过放）
          - 权重 10 > 5：过热比过冷更不可接受

        参数:
            T:   温度 [°C]
            SoC: 荷电状态 [0-1]

        返回:
            惩罚值 [元]
        """
        penalty = 0.0
        # ---- 温度惩罚（越出舒适区）----
        # 动态上限 T_hi = min(舒适上限35, 动态上限 38−4×SoC)
        #   SoC 越高，允许的温度上限越低（满电更怕热，见 thermal_model.dynamic_T_max）
        T_hi = min(self.T_target_high, self.T_dyn_base - self.T_dyn_slope * SoC)
        # 过冷惩罚（低于 25°C）：5 × (25 − T)² —— 权重 5，较轻（过度冷却只是浪费电）
        if T < self.T_target_low:
            penalty += 5.0 * (self.T_target_low - T) ** 2
        # 过热惩罚（高于 T_hi）：10 × (T − T_hi)² —— 权重 10，过热比过冷更不可接受
        elif T > T_hi:
            penalty += 10.0 * (T - T_hi) ** 2
        # ---- SoC 惩罚（避免过充/过放）----
        # 过放惩罚 20 × (0.15−SoC)²；过充惩罚 20 × (SoC−0.85)² —— 权重 20，最重
        if SoC < 0.15:
            penalty += 20.0 * (0.15 - SoC) ** 2
        elif SoC > 0.85:
            penalty += 20.0 * (SoC - 0.85) ** 2
        return penalty
    
    # def optimize(self, weather_hourly, price_hourly, T_initial,
    #              model: DetailedThermalModel, SoC_initial=0.5):

    def optimize(self,weather_hourly,price_hourly,T_initial,
                  model:DetailedThermalModel,SoC_initial=0.5,prev_a_initial=0,
                  start_hour=0,charge_latch_initial=None,P_target=None):
        """
        动态规划求解最优冷却计划（四维状态空间：温度 × SoC × 前冷却状态 × 充电锁存）。

        参数:
            weather_hourly: 天气数据（窗口相对序列，index 0 = 窗口第一步）
                {"temperature": ndarray[24], "radiation": ndarray[24]}
            price_hourly:   逐时电价 [元/kWh], ndarray[24]（窗口相对序列）
            T_initial:      当前电解液温度 [°C]
            model:          DetailedThermalModel 实例
            SoC_initial:    初始 SoC [0-1]（默认 0.5）
            prev_a_initial = 优化窗口开始前一小时的冷却状态
            （0=假设之前是关的，1=冷却正在运行中。）
            start_hour:     窗口起始钟表时刻（0-23，默认 0 = 日历日）。
            窗口第 h 步的钟表时刻为 (start_hour + h) % 24，充放电窗口、
            TOU、太阳辐射窗、锁存清除窗均按该钟表时刻判断。
            charge_latch_initial: 窗口开始时 BMS 充电锁存状态（0/1）。
            None（默认）= 按 SoC_initial 推导（向后兼容）；滚动运行时
            必须显式传入上一运行的真实锁存——锁存置位后 shunt 会把 SoC
            拖到 SoC_max 以下，靠 SoC 反推会错判为未锁。

        返回:
            result: 字典（与 simulate_24h 返回格式一致）
        """
        t_start = time_module.time()

        H = self.horizon_h
        nT = self.n_T
        nS = self.n_SoC
        T_grid = self.T_grid
        SoC_grid = self.SoC_grid

        assert len(price_hourly) >= H, \
            f"电价数据长度 {len(price_hourly)} < 优化时域 {H}"
        assert len(weather_hourly["temperature"]) >= H, \
            f"天气数据长度不足 {H}"

        # ================================================================
        # 初始化四维 DP 表
        # ================================================================


        ## ---- 状态空间剪枝：前向可达集（大幅降低 DP 计算量）----
        # 原理：从起点出发，物理上 24h 内"能到达"的温度/SoC 组合只是完整网格的一小片
        # （温度 1h 最多变 ~1°C、SoC 充电 1h 最多抬 ~0.14），其余大部分网格状态根本
        # 走不到。所以先正向推一遍"可达集"，后向递推时只算这些状态，省掉大量无谓调用。

        # 起点：初始 (温度, SoC) 量化到网格上的格点
        t0 = self._nearest_T_idx(T_initial)
        s0 = self._nearest_SoC_idx(SoC_initial)
        reach = [set() for _ in range(H+1)]   # reach[h] = 第 h 步所有可达的 (温度格, SoC格) 集合
        reach[0].add((t0, s0))
        # 逐小时正向扩展：对每个可达状态，试两种锁存 l × 两种动作 a，用物理模型推一步
        for hh in range(H):
            d_h = day_hour_of(start_hour, hh)
            T_a = float(weather_hourly["temperature"][hh])
            rad = float(weather_hourly["radiation"][hh])
            pr_h = float(price_hourly[hh])
            for (t, s) in list(reach[hh]):
                for l in (0, 1):    # 锁存遍历 0/1（过近似：宁多勿漏）
                    cs = model.determine_charge_discharge_state(
                        d_h, float(SoC_grid[s]), charge_latch=l,
                        P_target=(P_target[hh] if P_target is not None else None))
                    for a in (0, 1):   # 冷却动作：0 关 / 1 开
                        st = {"T_ely": float(T_grid[t]), "T_stack": float(T_grid[t]),
                              "SoC": float(SoC_grid[s]), "Current_State": cs,
                              "fan_state": 0, "charge_latch": l}
                        ns, _ = model.step_hourly(
                            st, d_h, {"temperature": T_a, "radiation": rad},
                            pr_h, cool_on=bool(a), heat_on=False)
                        # 把物理模型推出的下一步状态量化后，加入下一小时的可达集
                        reach[hh+1].add((self._nearest_T_idx(ns["T_ely"]),
                                         self._nearest_SoC_idx(ns["SoC"])))
        # 膨胀 ±1 格：前向追踪走的是"连续轨迹"，与这里"量化轨迹"可能有 1 格的取整漂移，
        # 所以在温度/SoC 两维各扩 1 格，保证不会把最优轨迹误剪掉
        dil = [set() for _ in range(H+1)]
        for hh in range(H+1):
            for (t, s) in reach[hh]:
                for dt in (-1, 0, 1):
                    for ds in (-1, 0, 1):
                        tt, ss = t+dt, s+ds
                        if 0 <= tt < nT and 0 <= ss < nS:
                            dil[hh].add((tt, ss))
        reach_list = [sorted(x) for x in dil]   # 排序成 list，后向递推按这个遍历
        

        # V[h][t][s][p][l] = 从第 h 步状态 (T_grid[t], SoC_grid[s]) 出发、
        #   上一小时冷却状态为 p、BMS 充电锁存为 l 时的最小剩余总费用
        V = np.full((H + 1, nT, nS, 2, 2), np.inf)
        # policy[h][t][s][p][l] = 第 h 步状态 (t, s, p, l) 的最优决策 (0=关, 1=开)
        policy = np.zeros((H, nT, nS, 2, 2), dtype=int)

        # 终值 = 温度偏离惩罚 + SoC 偏离惩罚
        # （两个 l 切片相同：终态锁存对后续成本无影响）
        for t in range(nT):
            for s in range(nS):
                pen = self._terminal_penalty(T_grid[t], SoC_grid[s])
                for p in range(2):
                    for l in range(2):
                        V[H, t, s, p, l] = pen

        # ================================================================
        # 第一阶段：向后递推（Backward Induction）
        # 状态 = (温度, SoC, 前冷却状态 p, 充电锁存 l)
        # SoC 动态变化影响充放电状态判断
        # ================================================================

        for h in range(H - 1, -1, -1):
            # 窗口第 h 步的钟表时刻（滚动时域；start_hour=0 时等于 h）
            day_h = day_hour_of(start_hour, h)

            T_a = float(weather_hourly["temperature"][h])
            rad = float(weather_hourly["radiation"][h])
            price = float(price_hourly[h])

            # for t in range(nT):
            for (t,s) in reach_list[h]:


                    T_ely = float(T_grid[t])
             
                # for s in range(nS):
                    SoC_val = float(SoC_grid[s])

                    for l in range(2):            # BMS 充电锁存 (0=未锁, 1=已锁)
                        # 充放电状态依赖 (钟表时刻, SoC, 锁存)，在 l 层计算一次
                        # （不依赖 p/a，无需下沉）
                        Current_State = model.determine_charge_discharge_state(
                            day_h, SoC_val, charge_latch=l,P_target=(P_target[h] if P_target is not None else None))

                        for p in range(2):        # 上一小时冷却状态
                            best_cost = np.inf
                            best_a = 0
                            for a in [0, 1]:
                                # 构造状态（使用网格上的 SoC 值，而非固定 0.5）
                                state = {
                                    "T_ely":         T_ely,
                                    "T_stack":       T_ely,     # 近似：堆温≈液温
                                    "SoC":           SoC_val,
                                    "Current_State": Current_State,
                                    "fan_state":     0,
                                    "charge_latch":  l,
                                }
                                weather_k = {"temperature": T_a, "radiation": rad}

                                # 调用热力学模型
                                next_state, metrics = model.step_hourly(
                                    state, day_h, weather_k, price,
                                    cool_on=bool(a), heat_on=False,

                                    #######等待接入真数据打开测试############
                                    #   P_target=(P_target[h] if P_target is not None else None)
                                      )

                                # ---- 即时费用 immediate（本步的代价）= 三部分之和 ----
                                # ① metrics['cost']：本步冷却电费 [元]（物理模型已算好）
                                # ② temp_penalty：本步末温度越限惩罚 [元]（下面算）
                                # ③ switch_cost：启停惩罚（本步动作 a 与上步 p 不同时罚 3 元）
                                T_next = next_state["T_ely"]
                                temp_penalty = 0.0
                                # 动态上限 T_hi = min(35, 38−4×SoC)：SoC 越高上限越严
                                T_hi = min(self.T_target_high,
                                           self.T_dyn_base - self.T_dyn_slope * SoC_val)
                                if T_next > T_hi:
                                    # 过热惩罚：10 × (T − T_hi)²
                                    temp_penalty = 10.0 * (T_next - T_hi) ** 2
                                elif T_next < self.T_target_low:
                                    # 过冷惩罚：5 × (25 − T)²
                                    temp_penalty = 5.0 * (self.T_target_low - T_next) ** 2

                                # 启停惩罚：状态翻转（关→开 或 开→关）一次罚 startup_penalty(3元)
                                switch_cost = self.startup_penalty if a != p else 0.0

                                immediate = metrics['cost'] + switch_cost + temp_penalty

                                # 查找下一步状态索引（四维）
                                t_next = self._nearest_T_idx(next_state["T_ely"])
                                s_next = self._nearest_SoC_idx(next_state["SoC"])
                                l_next = int(next_state["charge_latch"])

                                # Bellman 方程：总代价 = 即时费用 + 未来最优代价
                                # 公式：total = immediate + V[h+1][t_next][s_next][a][l_next]
                                #   （本步决策 a 会成为下一步的"上一小时冷却状态" p' = a）
                                total = immediate + V[h + 1, t_next, s_next, a, l_next]

                                if total < best_cost:
                                    best_cost = total
                                    best_a = a

                            V[h, t, s, p, l] = best_cost
                            policy[h, t, s, p, l] = best_a



        # ================================================================
        # 第二阶段：前向追踪（Forward Trace）
        # 从初始状态 (T_initial, SoC_initial) 出发，查四维 policy 表
        # ================================================================

        schedule = np.zeros(H, dtype=int)
        T_forecast = np.zeros(H + 1)
        T_stack_forecast = np.zeros(H + 1)
        SoC_forecast = np.zeros(H + 1)
        P_cool_arr = np.zeros(H)
        Q_cool_arr = np.zeros(H)
        P_pump_arr = np.zeros(H)
        P_fan_arr = np.zeros(H)
        P_heat_arr = np.zeros(H)
        P_agit_arr = np.zeros(H)
        P_flush_arr = np.zeros(H)
        Q_gen_arr = np.zeros(H)
        Q_wall_arr = np.zeros(H)
        Q_shunt_arr = np.zeros(H)
        cost_arr = np.zeros(H)
        COP_arr = np.zeros(H)
        metrics_all = []

        T_forecast[0] = T_initial
        T_stack_forecast[0] = T_initial
        SoC_forecast[0] = SoC_initial
        t_idx = self._nearest_T_idx(T_initial)
        s_idx = self._nearest_SoC_idx(SoC_initial)
        p_idx = prev_a_initial
        # 初始充电锁存：显式传入优先（滚动运行跨运行接续），
        # 否则按 SoC_initial 推导（日前链路向后兼容）
        if charge_latch_initial is None:
            l_idx = model.initial_charge_latch(SoC_initial)
        else:
            l_idx = int(charge_latch_initial)
        state = {
            "T_ely":         T_initial,
            "T_stack":       T_initial,
            "SoC":           SoC_initial,
            "Current_State": 0,
            "fan_state":     0,
            "charge_latch":  l_idx,
        }

        # === 能量累加变量===
        E_cool_elec_kWh = 0.0
        E_pump_elec_kWh = 0.0
        E_pump_cool_kWh = 0.0
        E_pump_not_cool_kWh = 0.0
        E_flush_elec_kWh = 0.0
        E_heat_elec_kWh = 0.0
        E_agit_elec_kWh = 0.0
        E_fan_elec_kWh = 0.0
        E_charge_grid_kWh = 0.0
        E_discharge_grid_kWh = 0.0
        Cost_cool_CNY = 0.0
        Cost_fan_CNY = 0.0
        dt = 3600.0

        for h in range(H):
            # 窗口第 h 步的钟表时刻（滚动时域；start_hour=0 时等于 h）
            day_h = day_hour_of(start_hour, h)

            # 查四维最优决策
            a = policy[h, t_idx, s_idx, p_idx, l_idx]
            schedule[h] = a

            # 更新充放电状态（使用动态 SoC + BMS 充电锁存）
            state["Current_State"] = model.determine_charge_discharge_state(
                day_h, state["SoC"], state.get("charge_latch", 0),P_target=(P_target[h] if P_target is not None else P_target))
            

            weather_k = {
                "temperature": float(weather_hourly["temperature"][h]),
                "radiation":   float(weather_hourly["radiation"][h]),
            }

            # 执行一步
            next_state, metrics = model.step_hourly(
                state, day_h, weather_k, float(price_hourly[h]),
                cool_on=bool(a), heat_on=False,

                #######等待接入真数据打开测试############
                # P_target=(P_target[h] if P_target is not None else P_target)
                )

            # 记录结果
            T_forecast[h + 1] = next_state["T_ely"]
            T_stack_forecast[h + 1] = next_state["T_stack"]
            SoC_forecast[h + 1] = next_state["SoC"]
            P_cool_arr[h] = metrics["P_cool_elec"]
            Q_cool_arr[h] = metrics["Q_cool"]
            P_pump_arr[h] = metrics["P_pump_elec"]
            P_fan_arr[h] = metrics["P_fan_elec"]
            P_heat_arr[h] = metrics["P_heat_elec"]
            P_agit_arr[h] = metrics["P_agit_elec"]
            P_flush_arr[h] = metrics["P_flush_elec"]
            Q_gen_arr[h] = metrics["Q_gen"]
            Q_wall_arr[h] = metrics["Q_wall"]
            Q_shunt_arr[h] = metrics["Q_shunt"]
            cost_arr[h] = metrics["cost"]
            COP_arr[h] = metrics["COP_now"]
            metrics_all.append(metrics)

            # === 能量累加 ===
            E_cool_elec_kWh += metrics["P_cool_elec"] * dt / 3.6e6
            E_pump_elec_kWh += metrics["P_pump_elec"] * dt / 3.6e6
            E_pump_cool_kWh += metrics["pump_cool_share"] * dt / 3.6e6
            E_pump_not_cool_kWh += (metrics["P_pump_elec"] - metrics["pump_cool_share"]) * dt / 3.6e6
            E_flush_elec_kWh += metrics["P_flush_elec"] * dt / 3.6e6
            E_heat_elec_kWh += metrics["P_heat_elec"] * dt / 3.6e6
            E_agit_elec_kWh += metrics["P_agit_elec"] * dt / 3.6e6
            E_fan_elec_kWh += metrics["P_fan_elec"] * dt / 3.6e6
            E_charge_grid_kWh += metrics["P_elec_charge"] * dt / 3.6e6
            E_discharge_grid_kWh += metrics["P_elec_disch"] * dt / 3.6e6

            # 成本累加
            Cost_cool_CNY += (metrics["P_cool_elec"] + metrics["pump_cool_share"]) * dt / 3.6e6 * price_hourly[h]
            Cost_fan_CNY += metrics["P_fan_elec"] * dt / 3.6e6 * price_hourly[h]

            # 更新四维状态索引
            t_idx = self._nearest_T_idx(next_state["T_ely"])
            s_idx = self._nearest_SoC_idx(next_state["SoC"])
            l_idx = int(next_state["charge_latch"])
            state = next_state
            p_idx = a

        # === 后处理 ===
        # for arr in [T_forecast, T_stack_forecast, SoC_forecast, P_cool_arr,
        #             Q_cool_arr, P_pump_arr, P_fan_arr, P_heat_arr, P_agit_arr,
        #             P_flush_arr, Q_gen_arr, Q_wall_arr, Q_shunt_arr, COP_arr]:
        #     arr[-1] = arr[-2]

        total_cost = float(np.sum(cost_arr))
        solve_time = time_module.time() - t_start

        # === 能量统计 [kWh] ===
        E_internal_kWh = float(np.sum(Q_gen_arr) * dt / 3.6e6)
        E_shunt_kWh = float(np.sum(Q_shunt_arr) * dt / 3.6e6)
        E_cool_th_kWh = float(np.sum(Q_cool_arr) * dt / 3.6e6)
        E_heat_th_kWh = float(np.sum(P_heat_arr) * dt / 3.6e6)
        E_wall_kWh = float(np.sum(Q_wall_arr) * dt / 3.6e6)
        E_parasitic_total_kWh = (E_cool_elec_kWh + E_pump_elec_kWh + E_flush_elec_kWh
                                 + E_heat_elec_kWh + E_agit_elec_kWh + E_fan_elec_kWh)
        Cost_total_CNY = Cost_cool_CNY + Cost_fan_CNY

        # === 往返效率 RTE（Round-Trip Efficiency）===
        # 电堆级 RTE = 放电电量 ÷ 充电电量 × 100%（只看电堆本身，不含寄生功耗）
        # 公式：RTE_stack = E_discharge / E_charge × 100%
        RTE_stack = (E_discharge_grid_kWh / E_charge_grid_kWh * 100) if E_charge_grid_kWh > 0 else float("nan")
        # 系统级 RTE = 放电电量 ÷ (充电电量 + 全部寄生功耗) × 100%（把泵/冷机等也算进去）
        # 公式：RTE_system = E_discharge / (E_charge + E_parasitic) × 100%
        RTE_system = (E_discharge_grid_kWh / (E_charge_grid_kWh + E_parasitic_total_kWh) * 100
                      ) if (E_charge_grid_kWh + E_parasitic_total_kWh) > 0 else float("nan")

        # 运行时长
        Cool_h = float(np.sum(schedule))
        Agit_h = float(np.sum(P_agit_arr > 0)) * model.agit_duty  # 搅拌器按占空比折算
        Flush_h = float(np.sum(P_flush_arr > 0)) * model.flush_duty_cycle  # 冲洗泵按占空比折算
        Fan_h = float(np.sum(P_fan_arr > 0))

        result = {
            "schedule":      schedule,
            "T_forecast":    T_forecast,
            "T_stack":       T_stack_forecast,
            "SoC_forecast":  SoC_forecast,
            "P_cool_elec":   P_cool_arr,
            "Q_cool":        Q_cool_arr,
            "P_pump_elec":   P_pump_arr,
            "P_fan_elec":    P_fan_arr,
            "P_heat_elec":   P_heat_arr,
            "P_agit_elec":   P_agit_arr,
            "P_flush_elec":  P_flush_arr,
            "Q_gen":         Q_gen_arr,
            "Q_wall":        Q_wall_arr,
            "Q_shunt":       Q_shunt_arr,
            "cost_hourly":   cost_arr,
            "total_cost":    total_cost,
            "COP_hourly":    COP_arr,
            "metrics_all":   metrics_all,
            "solve_time_s":  solve_time,
            "E_internal_kWh": E_internal_kWh,
            "E_shunt_kWh": E_shunt_kWh,
            "E_cool_th_kWh": E_cool_th_kWh,
            "E_heat_th_kWh": E_heat_th_kWh,
            "E_wall_kWh": E_wall_kWh,
            "E_cool_elec_kWh": E_cool_elec_kWh,
            "E_pump_elec_kWh": E_pump_elec_kWh,
            "E_pump_cool_kWh": E_pump_cool_kWh,
            "E_pump_not_cool_kWh": E_pump_not_cool_kWh,
            "E_flush_elec_kWh": E_flush_elec_kWh,
            "E_heat_elec_kWh": E_heat_elec_kWh,
            "E_agit_elec_kWh": E_agit_elec_kWh,
            "E_fan_elec_kWh": E_fan_elec_kWh,
            "E_parasitic_total_kWh": E_parasitic_total_kWh,
            "E_charge_grid_kWh": E_charge_grid_kWh,
            "E_discharge_grid_kWh": E_discharge_grid_kWh,
            "RTE_stack": RTE_stack,
            "RTE_system": RTE_system,
            "Cost_cool_CNY": Cost_cool_CNY,
            "Cost_fan_CNY": Cost_fan_CNY,
            "Cost_total_CNY": Cost_total_CNY,
            "Cool_h": Cool_h,
            "Agit_h": Agit_h,
            "Flush_h": Flush_h,
            "Fan_h": Fan_h,
        }

        return result

