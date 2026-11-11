# ============================================================================
# cooling_scheduler/thermal_model.py
# 模块 3：详细热力学模型（计划层）
#
# 职责：实现完整的热力学模型（不简化物理方程），以 1 小时为步长推进，
#       用于 DP 优化器的状态转移计算。
#
# 与原始 run_simulation() 的对应关系：
#   步骤1:  环境温度+太阳辐射     
#   步骤2:  电价查询               
#   步骤3:  通风风扇控制            
#   步骤4:  内部空气温度           
#   步骤5:  冷却控制状态机          
#   步骤6:  加热控制                
#   步骤7:  搅拌器                  
#   步骤8:  冲洗控制                
#   步骤9:  充放电状态机            
#   步骤10: 阿伦尼乌斯温度修正      
#   步骤11: 泵功耗                  
#   步骤12: 电化学计算             
#   步骤13: 冷却功率               
#   步骤14: 加热功率                
#   步骤15: 电堆温度               
#   步骤16: 电解液温度              
# 依赖：模块 1（config）
# 被依赖：模块 4（基线规则）、模块 5（DP 优化器）
# ============================================================================

import numpy as np
import math

from .config import SystemConfig


def day_hour_of(start_hour, k):
    """窗口相对步 k（0..H-1）→ 钟表时刻（0-23）。

    滚动时域窗口的起点可以是任意小时：窗口第 k 步对应的真实钟表
    时刻为 (start_hour + k) % 24。start_hour=0 时为恒等映射，
    行为与旧版本逐位一致（日前日历日链路）。
    """
    return (int(start_hour) + k) % 24


class DetailedThermalModel:
    """
    集装箱式液流电池储能系统详细热力学模型（1 小时步长）。

    原始 run_simulation() 中的全部物理方程：
      - 完整热阻网络（UA 串联 → 化简）
      - 阿伦尼乌斯温度相关（电阻、黏度）
      - 电化学模型（Nernst OCV、欧姆热、熵热、旁路电流）
      - 冷却物理极限（UA × ΔT × LMTD）+ COP 温度衰减
      - 电堆热质量（金属 + 塑料 + 滞留电解液）

    """

    def __init__(self, config: SystemConfig):
        """
        从 SystemConfig 初始化热力学模型。
        提取所有物理参数并预计算派生量。

        参数:
            config: SystemConfig 实例
        """
        self.config = config
        self.tech_id = config.tech_id
        self.dt = 3600.0        # 1 小时步长 [秒]

        # ================================================================
        # 电解液热容
        # ================================================================
        self.C_th = config.phys["C_th"]                # [J/K] 总热容

        # ================================================================
        # 集装箱热阻网络
        # ================================================================
        self.UA_tank_inner = config.UA_tank_inner      # [W/K] 储罐→室内
        self.UA_wall_inner = config.UA_wall_inner       # [W/K] 外壳→室内
        self.UA_total = config.UA_total                 # [W/K] 总和

        # ================================================================
        # 太阳辐射参数
        # ================================================================
        self.alpha = config.solar["alpha"]              # 外壳吸收系数
        self.h_ext = config.solar["h_ext"]              # [W/m²K] 对流换热系数
        self.sunrise = config.solar["sunrise"]          # [小时] 日出
        self.sunset = config.solar["sunset"]            # [小时] 日落
        self.day_len_h = config.solar["day_len_h"]      # [小时] 日照时长

        # ================================================================
        # 电堆参数
        # ================================================================
        self.P_nom = config.stack["P_nom"]              # [W] 额定功率
        self.Cap_nom = config.stack["Cap_nom"]          # [Wh] 额定容量
        self.SoC_min = config.stack["SoC_min"]
        self.SoC_max = config.stack["SoC_max"]
        self.R_ref = config.stack["R_ref"]              # [Ω] 25°C 基准内阻
        self.Ea_R = config.stack["Ea_R"]                # 电阻活化能
        self.U_OCV_ref = config.stack["U_OCV_ref"]      # [V] 基准开路电压
        self.dS_coeff = config.stack["dS_coeff"]        # 熵热系数
        self.P_shunt_W = config.stack["P_shunt_W"]      # [W] 旁路损耗系数
        self.start_charge_h = config.stack["start_charge_h"]
        self.start_disch_h = config.stack["start_disch_h"]
        self.end_charge_h = 17.0                # [小时] 充电窗口结束时刻（单点定义）

        # ================================================================
        # 泵参数
        # ================================================================
        self.P_pump_rated = config.pump["P_rated"]      # [W] 额定泵功率
        self.P_pump_idle = config.pump["P_idle"]        # [W] 怠速泵功率
        self.Ea_visc = config.pump["Ea_visc"]           # 黏度活化能

        # ================================================================
        # 换热器参数
        # ================================================================
        self.UA_hex = config.hex["plate_UA"]            # [W/K]
        self.T_cool_in = config.hex["plate_T_in"]       # [°C] 冷源入口温度
        self.Q_cool_max = config.hex["plate_P_max"]     # [W] 冷却功率上限
        self.COP_base = config.hex["plate_COP"]         # 基准 COP
        self.COP_deg = config.hex["COP_deg"]            # COP 衰减因子
        self.LMTD_corr = config.hex["LMTD_corr"]        # LMTD 修正系数

        # ================================================================
        # 加热参数
        # ================================================================
        self.heat_active = config.heat["active"]          # 是否启用加热
        self.P_heat_W = config.heat["P_heat_W"]         # [W] 加热功率
        self.COP_heat = config.heat["COP_heat"]         # 加热效率
        self.T_heat_on = config.heat["T_on"]            # [°C] 加热启动阈值
        self.T_heat_off = config.heat["T_off"]          # [°C] 加热停止阈值

        # ================================================================
        # 搅拌器参数
        # ================================================================
        self.P_agit_W = config.agitator["P_agit_W"]     # [W] 搅拌器功率
        self.agit_duty = config.agitator["duty_cycle"]   # 占空比

        # ================================================================
        # 电堆热质量
        # ================================================================
        self.UA_stack = config.stack_thermal["UA_stack"]         # [W/K]
        self.C_th_stack = config.stack_thermal["C_th_stack"]     # [J/K]

        # ================================================================
        # 通风参数
        # ================================================================
        self.vent_active = config.vent_active
        self.vent_P_fan = config.vent["P_fan"]                   # [W]
        self.vent_residual = config.vent["residual_factor"]
        self.vent_dT_on = config.vent["dT_trigger_on"]           # [K]
        self.vent_dT_off = config.vent["dT_trigger_off"]         # [K]

        # ================================================================
        # 冲洗估算参数
        # ================================================================
        # 冲洗占空比：每次冲洗 120s，按每小时触发 1 次估算
        flush_duration_s = config.flush_cfg["duration_s"]  # 120 s
        self.flush_duty_cycle = flush_duration_s / self.dt  # 120/3600 = 1/30
        self.P_flush_est = config.pump["P_idle"] * self.flush_duty_cycle  # [W] 估算平均冲洗功率
        self.T_flush_trigger = config.stack_thermal["T_flush_trigger"]  # [°C] 冲洗触发温度

        # ================================================================
        # 旁路功率
        # ================================================================
        self.P_shunt_coeff = config.stack["P_shunt_W"]           # [W] per SoC

        self.T_dyn_base=config.optimizer.get("T_dyn_base",38.0)   #SoC=0时的上限
        self.T_dyn_slope=config.optimizer.get("T_dyn_slope",4.0)    #每单位SoC下降幅度

    #满电上限更严、空电放宽：SoC=0.9→34.4°C, SoC=0.1→37.6°C
    def dynamic_T_max(self,SoC):
        return self.T_dyn_base-self.T_dyn_slope*float(SoC)

    # ================================================================
    # 核心方法：单步热力学计算
    # ================================================================

    def step_hourly(self, state, hour, weather, price, cool_on, heat_on=False,P_target=None):
        """
        单步热力学计算（1 小时步长）。
        仅将冷却控制（步骤5）替换为外部决策变量。

        参数:
            state:   当前系统状态字典
                     {
                       "T_ely": float,       # 电解液温度 [°C]
                       "T_stack": float,     # 电堆温度 [°C]
                       "SoC": float,         # 荷电状态 [0-1]
                       "Current_State": int, # 电堆状态 0=待机/1=充电/2=放电
                       "fan_state": int,     # 风扇状态 0=关/1=开
                       "charge_latch": int,  # BMS 充电锁存 0=未锁/1=已锁
                                             #（可选，缺省按 0 处理）
                     }
            hour:    当前时刻（0-23 的十进制小时数）
            weather: 天气数据字典
                     {
                       "temperature": float,  # 环境温度 [°C]
                       "radiation": float,    # 短波辐照度 [W/m²]
                     }
            price:   当前电价 [元/kWh]
            cool_on: 冷却是否开启（优化器决策变量）bool
            heat_on: 加热是否开启（可选，默认 False）bool

        返回:
            (next_state, metrics)

            next_state: 下一步系统状态字典（同 state 结构）
            metrics:    性能指标字典
                        {
                          "Q_cool": float,        # 冷却热功率 [W]
                          "P_cool_elec": float,   # 冷却电功率 [W]
                          "P_pump_elec": float,   # 泵电功率 [W]
                          "P_flush_elec": float,  # 冲洗泵电功率 [W]
                          "P_fan_elec": float,    # 风扇电功率 [W]
                          "P_heat_elec": float,   # 加热电功率 [W]
                          "P_agit_elec": float,   # 搅拌器电功率 [W]
                          "Q_gen": float,         # 总内部产热 [W]
                          "Q_wall": float,        # 壁面传热 [W]
                          "Q_stack": float,       # 电堆产热 [W]
                          "Q_shunt": float,       # 旁路电流热 [W]
                          "COP_now": float,       # 当前 COP
                          "visc_factor": float,   # 黏度因子
                          "total_elec_W": float,  # 总寄生电功率 [W]
                          "pump_cool_share": float, # 分配给冷却的泵功率 [W]
                          "cost": float,          # 本步电费 [元]
                        }
        """
        # 提取当前状态
        T_ely = state["T_ely"]
        T_stack = state["T_stack"]
        SoC = state["SoC"]
        Current_State = state["Current_State"]
        fan_state = state.get("fan_state", 0)
        charge_latch = int(state.get("charge_latch", 0))   # BMS 充电锁存（缺省未锁）

        T_air = weather["temperature"]
        radiation = weather["radiation"]

        # ================================================================
        # 步骤 1：环境温度 + 太阳辐射
        # ================================================================

        # 太阳辐射引起的等效温升 dT_solar [K]
        # 公式：dT_solar = (α × I) / h_ext
        #   α(alpha)=0.6  —— 外壳对太阳的吸收系数（无量纲，深色壳吸收多）
        #   I=radiation   —— 短波辐照度 [W/m²]（Open-Meteo 的 GHI 水平面总辐照）
        #   h_ext=35      —— 外壳对流传热系数 [W/m²K]
        # 物理含义：太阳晒在外壳上，等效于把外界温度抬高了 dT_solar
        dT_solar = 0.0
        if self.sunrise <= hour <= self.sunset and self.day_len_h > 0:
            # 只有白天（日出 5.75h ~ 日落 19.17h）才有太阳辐射
            I_now = self.alpha * radiation  # 外壳吸收到的辐射热流 = α × I [W/m²]
            dT_solar = I_now / self.h_ext   # 辐射温升 = 热流 ÷ 对流换热系数 [K]

        # 综合温度（Sol-Air Temperature）Tsol-air = 环境温度 + 太阳辐射温升
        # 公式：T_sol_air = T_air + dT_solar
        # 这是"外墙感受到的有效外界温度"，太阳越强它越高
        T_sol_air = T_air + dT_solar

        # ================================================================
        # 步骤 3：通风风扇控制
        # 简化：逐小时滞回判断（不使用秒级计时器）
        # ================================================================

        if self.vent_active:
            # dT_wa = 太阳辐射造成的温升（用来判断通风风扇要不要启动）
            dT_wa = T_sol_air - T_air       # = dT_solar

            # 通风风扇滞回控制（防止频繁启停）：
            #   风扇开着、且辐射温升降到 < dT_trigger_off(1K) → 关
            #   风扇关着、且辐射温升升到 > dT_trigger_on(3K)  → 开
            if fan_state == 1 and dT_wa < self.vent_dT_off:
                fan_state = 0
            elif fan_state == 0 and dT_wa > self.vent_dT_on:
                fan_state = 1

            # 墙外感受到的边界温度 T_cont（用于算墙传热）
            # 风扇开 → 通风带走一部分太阳热，有效温升打 residual_factor(≈0.099) 折
            # 风扇关 → 墙外就是完整 sol-air 温度
            if fan_state:
                T_cont = T_air + self.vent_residual * dT_solar
            else:
                T_cont = T_sol_air
        else:
            T_cont = T_sol_air

        # ================================================================
        # 步骤 4：集装箱内部空气温度
        # 代数求解：室内空气节点热容忽略
        # T_innen = (UA_tank × T_ely + UA_wall × T_cont) / UA_total
        # ================================================================
        #集装箱内部空气温度=（（罐→箱内空气的热导*电解液温度+箱壁→箱内空气的热导*箱壁外侧"综合温度"）/两条热导之和）
        T_innen = ((self.UA_tank_inner * T_ely + self.UA_wall_inner * T_cont)
                   / self.UA_total)

        # ================================================================
        # 步骤 13：冷却功率计算
        # 完全保留：COP 温度衰减 + 物理极限（UA × ΔT × LMTD）
        # ================================================================

        Q_cool = 0.0
        P_cool_elec = 0.0
        COP_now = 0.0

        if cool_on:
            dT_amb = T_air - 25.0                   # 环境温度与参考温度差值

            if self.tech_id == 1:
                # ========================================================
                # 板式换热器 + 外部冷水机组（间接回路）
               
                # ========================================================

                # COP（性能系数）= 抽走的热量 ÷ 耗电量，越大越省电

                #   COP_base=2.9（25°C 基准）；COP_deg=0.08（环境每高 1°C 掉 0.08）
                #   环境越热，冷机越难把热"搬"出去，效率越低
                COP_now = max(self.COP_base - self.COP_deg * dT_amb, 1.5)

                # 换热器温差：电解液温度 − 冷源(7°C)温度 [K]
                dT_hex = T_ely - self.T_cool_in
                if dT_hex > 0:
                    # 物理极限制冷功率 Q_physics [W] = UA × ΔT × LMTD修正
                    #   UA_hex=544 [W/K]；LMTD_corr=0.9（对数平均温差修正）
                    Q_physics = self.UA_hex * dT_hex * self.LMTD_corr
                else:
                    Q_physics = 0.0    # 电解液比冷源还冷时无法制冷

                # 实际制冷量 = min(物理极限, 冷机容量上限 14kW)
                Q_cool = min(Q_physics, self.Q_cool_max)

                # 冷却电功率 = 制冷量 ÷ COP（同样的冷量，COP 越高越省电）
                if COP_now > 0:
                    P_cool_elec = Q_cool / COP_now

            else:
                # ========================================================
                # EKES 管束直接冷却（主电解液回路）
                
                # ========================================================

                # 容量因子 cap_f：电解液温度低于 22°C 时制冷能力急剧下降
                #   即 22°C 以下停冷，32°C 以上满冷，中间线性过渡
                cap_f = min(1.0, max(0.0, T_ely - 22.0) / 10.0)

                # 环境降容因子 derating：环境越热(>25°C)，制冷能力线性衰减
                if dT_amb > 0:
                    derating = max(1.0 - 0.005 * dT_amb, 0.0)
                else:
                    derating = 1.0

                # 实际制冷量 = 额定制冷量 × 环境降容 × 温度容量因子（且 ≤ 额定值）
                Q_cool = float(np.clip(
                    self.config.hex["ekes_Q_nom"] * derating * cap_f,
                    0.0, self.config.hex["ekes_Q_nom"]))

                # EKES 的 COP：基准 3.65，环境每高 1°C 掉 0.05，下限 2.0
                COP_now = max(self.config.hex["ekes_COP_nom"]
                              - 0.05 * dT_amb, 2.0)

                # 冷却电功率：COP 模式用 Q/COP；否则按部分负载线性模型
                if Q_cool > 0 and COP_now > 0:
                    if self.config.ekes_power_model == "COP":
                        P_cool_elec = Q_cool / COP_now
                    else:
                        # 部分负载功耗模型：P = P_nom × (0.2 + 0.8 × Q/Q_nom)
                        P_cool_elec = (self.config.hex["ekes_P_elec_nom"]
                                       * (0.2 + 0.8 * Q_cool
                                          / self.config.hex["ekes_Q_nom"]))

        # ================================================================
        # 步骤 9 + 10 + 12：充放电 + 阿伦尼乌斯 + 电化学
        
        # 简化：按固定时段判断 Current_State，不做秒级状态切换
        # ================================================================

        T_K = T_ely + 273.15                            # 绝对温度 [K]

        # 电堆内阻 R_stack [Ω]：Arrhenius 温度相关
        # 公式：R = R_ref × exp(Ea_R × (1/T − 1/298.15))
        #   R_ref=0.015Ω（25°C 基准）；Ea_R=2500；T 越高电阻越小
        R_stack = self.R_ref * np.exp(
            self.Ea_R * (1.0 / T_K - 1.0 / 298.15))

        # 安全的 SoC 范围（Nernst 公式在 0/1 附近对数会发散，先夹到 [0.05, 0.95]）
        SoC_safe = float(np.clip(SoC, 0.05, 0.95))

        # 开路电压 U_OCV [V]：类 Nernst 方程对数近似
        # 公式：U_OCV = 55 + 1.5 × ln(SoC / (1 − SoC))
        #   SoC 越高电压越高；SoC=0.5 时对数项=0，正好 55V
        U_OCV = self.U_OCV_ref + 1.5 * math.log(SoC_safe / (1.0 - SoC_safe))

        # 最大放电功率（电源内阻匹配极限）
        # 公式：P_max = U² / (4R) —— 外接负载电阻 = 内阻时输出功率最大
        P_max_disch = U_OCV ** 2 / (4.0 * R_stack)

        # ---- 部分负载：有外部调度目标(P_target)就用目标幅值，否则用额定 ----
        #   P_target 约定：>0 放电 / <0 充电 / =0 待机，单位 [W]
        #   有外部调度时，充/放电功率 = 目标幅值（而不是死板的额定 45kW）
        if P_target is not None:
            P_part=abs(float(P_target))             # 取功率幅值（正数）
            P_chg=min(P_part,self.P_nom)            # 充电功率（≤ 额定 45kW）
            P_dis=min(P_part,P_max_disch*0.95)      # 放电功率（≤ 最大放电×95% 留裕度）
        else:
            P_chg=self.P_nom                        # 无外部目标 → 按额定功率充
            P_dis=min(self.P_nom,P_max_disch*0.95)  # 额定放电（含 5% 裕度）
        


        # 实际放电功率，留 5% 裕度
        # P_elec_disch = min(self.P_nom, P_max_disch * 0.95)
        P_elec_disch=P_dis


        Q_stack_total = 0.0                             # 电堆总产热 [W]
        SoC_next_raw = SoC                              # 原始 SoC 下一步值
        f = 1.0                                         # 充放电活跃时间分数（边界截断用）
        clipped_charge = False                          # 充电撞限截断标志（BMS 锁存置位用）

        if Current_State == 1:
            # ============================================================
            # 充电
            # ============================================================
            # 电流计算：由 P = U×I 和 U = U_OCV + I×R 推导
            # I = (-U_OCV + sqrt(U_OCV² + 4×R×P)) / (2×R)
            I_A = ((-U_OCV + math.sqrt(max(0.0,
                         U_OCV ** 2 + 4 * R_stack * P_chg)))
                   / (2 * R_stack))

            # 电堆产热 = 欧姆热 - 熵热（充电时熵热为吸热）
            Q_stack_full = I_A ** 2 * R_stack - I_A * T_K * self.dS_coeff

            # SoC 增量：法拉第积分（用基准电压 55V 近似计量）
            # 公式：ΔSoC = (I × 55V × Δt) / Cap_nom，Δt = 3600s，Cap_nom=270000Wh
            dSoC_full = (I_A * self.U_OCV_ref
                         * self.dt / 3600.0) / self.Cap_nom

            # ---- 边界截断：若本步会撞穿 SoC_max，只算撞到限幅前的活跃时间 ----
            # 真实 BMS 到达 SoC_max 即停充，剩余时间不产热、不改变 SoC
            headroom = self.SoC_max - SoC
            clipped_charge = (dSoC_full > 0 and headroom < dSoC_full)
            if clipped_charge:
                f = max(0.0, headroom / dSoC_full)   # 活跃时间分数 (0,1]
            else:
                f = 1.0
            Q_stack_total = Q_stack_full * f
            SoC_next_raw = SoC + dSoC_full * f

        elif Current_State == 2:
            # ============================================================
            # 放电
            # ============================================================
            # 电流计算
            disc = max(0.0, U_OCV ** 2 - 4 * R_stack * P_elec_disch)
            I_A = (U_OCV - math.sqrt(disc)) / (2 * R_stack)

            # 电堆产热 = 欧姆热 + 熵热（放电时熵热为放热）
            Q_stack_full = I_A ** 2 * R_stack + I_A * T_K * self.dS_coeff

            # SoC 增量（放电为负，SoC 减少）：法拉第积分，公式同充电
            # 公式：ΔSoC = (I × 55V × Δt) / Cap_nom
            dSoC_full = (I_A * self.U_OCV_ref
                         * self.dt / 3600.0) / self.Cap_nom

            # ---- 边界截断：若本步会跌破 SoC_min，只算跌破前的活跃时间 ----
            # 真实 BMS 到达 SoC_min 即停放，剩余时间不产热、不改变 SoC
            headroom = SoC - self.SoC_min
            if dSoC_full > 0 and headroom < dSoC_full:
                f = max(0.0, headroom / dSoC_full)   # 活跃时间分数 (0,1]
            else:
                f = 1.0
            Q_stack_total = Q_stack_full * f
            SoC_next_raw = SoC - dSoC_full * f

        # 旁路电流损耗（自放电产热）：与 SoC 成正比
        # 公式：Q_shunt = P_shunt(450W) × SoC  [W] —— SoC 越高自放电越强
        Q_shunt = self.P_shunt_coeff * SoC

        # 自放电导致的 SoC 损失（把热量换算成电量）
        # 公式：ΔSoC_loss = Q_shunt × Δt / Cap_nom
        SoC_loss = Q_shunt * self.dt / 3600.0 / self.Cap_nom

        # 更新 SoC = 充/放电后的值 − 自放电损失，夹到 [0,1]
        SoC_next = float(np.clip(SoC_next_raw - SoC_loss, 0.0, 1.0))

        # ================================================================
        # 步骤 10：黏度因子
        # ================================================================

        # 黏度因子 visc_factor：电解液黏度随温度变化，影响泵的耗电
        # 公式：μ = exp(Ea_visc × (1/T − 1/298.15))，夹到 [0.5, 3.0]
        #   温度越低电解液越黏稠，泵越费力（泵功率 = 负载 × 黏度因子）
        visc_factor = float(np.clip(
            np.exp(self.Ea_visc * (1.0 / T_K - 1.0 / 298.15)),
            0.5, 3.0))

        # ================================================================
          
        # ================================================================

        if Current_State != 0:
            # 充放电期间：主泵以额定功率运行
            pump_load = self.P_pump_rated
            # ---- 边界截断联动：充放电只活跃了 f 比例的时间，泵同比例运行 ----
            # （EKES 直冷同时运行时泵需为冷却全程运行，不缩放）
            if f < 1.0 and not (self.tech_id == 2 and cool_on):
                pump_load *= f
        elif self.tech_id == 1:
            # 板式换热器（间接冷却）：待机冷却时主泵不需要运行
            pump_load = 0.0
        else:
            # EKES（直接冷却）：冷却运行时主泵必须运行
            pump_load = self.P_pump_rated if cool_on else 0.0

        # 实际泵功耗 = 负载 × 黏度因子（温度越低黏度越大、越费电）
        q_pumps = pump_load * visc_factor

        # 分配给冷却的泵功率：仅 tech_id=2 且"待机+冷却"时，泵功耗计入冷却成本
        #（板式冷机待机冷却不经过主回路，泵功耗不计入冷却）
        if self.tech_id == 2 and cool_on and Current_State == 0:
            pump_cool_share = q_pumps
        else:
            pump_cool_share = 0.0

        # 冲洗泵估算（改进：根据冲洗触发条件计算，与 hot_model_process.py 对齐）
        # 冲洗条件：待机 + 未冷却 + 电堆温度低于触发阈值
        flush_triggered = (Current_State == 0 and not cool_on
                           and T_stack <= self.T_flush_trigger)

        if flush_triggered:
            # 估算冲洗功率：怠速功率 × 占空比（120s/3600s = 1/30）× 黏度因子
            P_flush = self.P_flush_est * visc_factor
        else:
            P_flush = 0.0

        # ================================================================
        # 步骤 12 续：总产热
        # Q_gen = Q_stack + q_pumps + P_flush + Q_shunt
        # ================================================================

        Q_gen = Q_stack_total + q_pumps + P_flush + Q_shunt

        # ================================================================
        # 步骤 14：加热功率
        # 互锁：冷却运行时禁止加热
        # ================================================================

        if heat_on and not cool_on:
            Q_heat = self.P_heat_W
            P_heat_elec = Q_heat / self.COP_heat
        else:
            Q_heat = 0.0
            P_heat_elec = 0.0

        # ================================================================
        # 步骤 7：搅拌器
        # 简化：按占空比折算平均功率
        # ================================================================

        agit_cond = (Current_State != 0) or heat_on or cool_on
        P_agit = self.P_agit_W * self.agit_duty if agit_cond else 0.0

        # ================================================================
        # 步骤 3 续：风扇功率
        # ================================================================

        P_fan = self.vent_P_fan if fan_state else 0.0

        # ================================================================
        # 步骤 15：电堆温度更新
        # 修复：添加冲洗时的强制热平衡判断
        # ================================================================

        # 估算冲洗状态：待机 + 未冷却 + 电堆温度低于触发阈值
        flush_running_est = (Current_State == 0 and not cool_on
                             and T_stack <= self.T_flush_trigger)

        # 泵运行判断：充放电 OR 冷却 OR 冲洗
        pumps_running = (Current_State != 0) or cool_on or flush_running_est

        if pumps_running:
            # 泵运行时：强制热平衡，电堆温度 = 电解液温度
            T_stack_next = T_ely
        else:
            # 泵停止时：牛顿冷却模型，电堆温度向室内空气温度衰减
            dT = (self.UA_stack * (T_innen - T_stack)
                  / self.C_th_stack) * self.dt
            T_stack_next = T_stack + dT

        # ================================================================
        # 步骤 16：电解液温度更新
        # T_ely(t+1) = T_ely(t) + (Q_gen + Q_wall - Q_cool + Q_heat) × dt / C_th
        # ================================================================

        # 壁面热流
        Q_wall = self.UA_tank_inner * (T_innen - T_ely)

        # 电解液温度更新
        T_ely_next = T_ely + ((Q_gen + Q_wall - Q_cool + Q_heat)
                              * self.dt / self.C_th)

        # ================================================================
        # 成本计算
        # ================================================================

        # 总寄生电功率 total_elec [W] = 所有辅助设备耗电之和
        #   冷机 + 主泵 + 冲洗泵 + 加热 + 搅拌 + 风扇
        total_elec = (P_cool_elec + q_pumps + P_flush
                      + P_heat_elec + P_agit + P_fan)

        # 本步冷却电费 cost [元] = (冷却电功率 + 分配给冷却的泵功率) × 电价
        # 公式：cost = (P_cool_elec + pump_cool_share) / 1000 × price
        #   ÷1000 是 W→kW 换算；price 单位 元/kWh；1 小时时长正好 = 1kWh 对应的电量
        # 注意：其他寄生功耗（充放电泵、搅拌器、风扇）不进冷却成本——
        #   它们由各自的时段决定，不随冷却计划(决策变量)变化，故不参与 DP 优化目标
        cost = (P_cool_elec + pump_cool_share) / 1000.0 * price

        # ================================================================
        # BMS 充电锁存更新
        # 充电窗口外：清除锁存（下一充电窗口开始时解锁）；
        # 充电窗口内：已锁保持；本步充电撞 SoC_max 截断时置位；
        # 入步 SoC 已 >= SoC_max 时同样置位（保证 DP 网格满电行自洽）。
        # ================================================================
        if self.start_charge_h <= hour < self.end_charge_h:
            latch_next = int(charge_latch
                             or (Current_State == 1 and clipped_charge)
                             or (SoC >= self.SoC_max))
        else:
            latch_next = 0

        # ================================================================
        # 构建输出
        # ================================================================

        next_state = {
            "T_ely":         T_ely_next,
            "T_stack":       T_stack_next,
            "SoC":           SoC_next,
            "Current_State": Current_State,
            "fan_state":     fan_state,
            "charge_latch":  latch_next,
        }

        metrics = {
            "Q_cool":          Q_cool,               # [W] 冷却热功率
            "P_cool_elec":     P_cool_elec,           # [W] 冷却电功率
            "P_pump_elec":     q_pumps,               # [W] 泵电功率
            "P_flush_elec":    P_flush,               # [W] 冲洗泵电功率
            "P_fan_elec":      P_fan,                 # [W] 风扇电功率
            "P_heat_elec":     P_heat_elec,           # [W] 加热电功率
            "P_agit_elec":     P_agit,                # [W] 搅拌器电功率
            "Q_gen":           Q_gen,                 # [W] 总内部产热
            "Q_wall":          Q_wall,                # [W] 壁面传热
            "Q_stack":         Q_stack_total,         # [W] 电堆产热
            "Q_shunt":         Q_shunt,               # [W] 旁路电流热
            "Q_heat":          Q_heat,                # [W] 加热热功率
            "COP_now":         COP_now,               # 当前 COP
            "visc_factor":     visc_factor,            # 黏度因子
            "total_elec_W":    total_elec,            # [W] 总寄生电功率
            "pump_cool_share": pump_cool_share,       # [W] 冷却分配泵功率
            "cost":            cost,                  # [元] 本步冷却电费
            # === 充放电功率（用于往返效率计算）===
            # 边界截断：只按活跃时间分数 f 计入充放电电量
            # "P_elec_charge":   self.P_nom * f if Current_State == 1 else 0.0,  # [W] 充电功率
            "P_elec_charge":   P_chg * f  if Current_State == 1 else 0.0,       # [W] 充电功率,
            "P_elec_disch":    P_elec_disch * f if Current_State == 2 else 0.0, # [W] 放电功率
        }

        return next_state, metrics

    # ================================================================
    # 辅助方法：充放电状态判断
    # ================================================================

    def determine_charge_discharge_state(self, hour, SoC, charge_latch=0,P_target=None):
        """
        根据当前时刻、SoC 和 BMS 充电锁存判断充放电状态。

        规则：
          充电时段：start_charge_h ~ end_charge_h (17:00)，且未锁存，且 SoC < SoC_max
          放电时段：start_disch_h ~ 次日 06:00，且 SoC > SoC_min

        充电锁存（charge_latch）：显式 BMS 充电锁，由 step_hourly() 维护。
          充电到达 SoC_max（边界截断）或入步即满时置位，此后同一充电
          窗口内禁止再充电（无论 shunt 自放电把 SoC 拖低多少，都不会
          出现涓流补电脉冲）；离开充电窗口后由 step_hourly() 清除锁存。
          取代旧版本的 eps=0.003 容差。
          放电侧无需锁存：放电截断后 shunt 把 SoC 推离 SoC_min，
          物理上不可能误重启放电。

        参数:
            hour:         当前时刻（0-23 的十进制小时数）
            SoC:          当前荷电状态 [0-1]
            charge_latch: BMS 充电锁存（0=未锁, 1=已锁禁止充电）

        返回:
            int: 0=待机, 1=充电, 2=放电
        """
        #外部调度：给了功率目标就用P_target,完全不用看硬编码时段
        if P_target is not None:
            return self._state_from_power(P_target,SoC,charge_latch)
        
        if (self.start_charge_h <= hour < self.end_charge_h
                and not charge_latch and SoC < self.SoC_max):
            return 1       # 充电
        elif (hour >= self.start_disch_h or hour < 6) and SoC > self.SoC_min:
            return 2       # 放电
        else:
            return 0       # 待机
    def _state_from_power(self,P_t,SoC,charge_latch=0):
        """由外部功率目标推到状态：>0放电 <0充电 =0待机（含安全裁剪）"""
        p=float(P_t)
        if p>0 and SoC >self.SoC_min:
            return 2
        if p<0 and not charge_latch and SoC <self.SoC_max:
            return 1
        return 0
    
    def initial_charge_latch(self, SoC):
        """
        由初始 SoC 推导初始充电锁存。

        已满电（SoC >= SoC_max）→ 锁存（1）；否则未锁（0）。
        simulate_24h / baseline / optimizer 前向追踪统一使用。

        参数:
            SoC: 初始荷电状态 [0-1]

        返回:
            int: 0=未锁, 1=已锁
        """
        return int(SoC >= self.SoC_max)

    # ================================================================
    # 辅助方法：加热状态机
    # ================================================================

    def determine_heating_state(self, T_ely, heat_is_running, cool_on):
        """
        根据电解液温度和冷却状态判断加热状态。
        完全复刻原始 hot_model_process.py 第 779-789 行的逻辑。

        控制逻辑：
          - 温度 < T_on (12°C)：启动加热
          - 温度 >= T_off (15°C)：停止加热
          - 冷却运行时：禁止加热（互锁）

        参数:
            T_ely:            当前电解液温度 [°C]
            heat_is_running:  当前加热运行状态 (0=停止, 1=运行)
            cool_on:          当前冷却是否运行

        返回:
            int: 0=停止加热, 1=运行加热
        """
        if not self.heat_active:
            return 0

        # 滞回控制：根据当前状态和温度决定下一状态
        if heat_is_running == 0 and T_ely < self.T_heat_on:
            # 当前停止，温度低于启动阈值 → 启动加热
            heat_is_running = 1
        elif heat_is_running == 1 and T_ely >= self.T_heat_off:
            # 当前运行，温度达到停止阈值 → 停止加热
            heat_is_running = 0

        # 互锁：冷却运行时禁止加热
        if cool_on:
            heat_is_running = 0

        return heat_is_running

    # ================================================================
    # 辅助方法：多步仿真
    # ================================================================

    def simulate_24h(self, weather_hourly, price_hourly, cooling_schedule,
                     T_initial=25.0, SoC_initial=0.5, heat_enabled=True,
                     start_hour=0,P_target=None):
        """
        用给定的冷却计划运行 24h 仿真。

        参数:
            weather_hourly:    24h 天气数据
                               {"temperature": array[24], "radiation": array[24]}
            price_hourly:      24h 电价 [元/kWh], array[24]
            cooling_schedule:  24h 冷却计划, array[24] (0/1)
            T_initial:         初始电解液温度 [°C]
            SoC_initial:       初始 SoC [0-1]
            heat_enabled:      是否启用加热状态机（默认 True）
            start_hour:        窗口起始钟表时刻（0-23，默认 0 = 日历日）。
                               滚动时域下窗口第 k 步的钟表时刻为
                               (start_hour + k) % 24；充放电窗口、TOU、
                               太阳辐射窗、锁存清除窗均按该钟表时刻判断。

        返回:
            result: 字典
                    {
                      "T_ely": array[25],         # 电解液温度轨迹
                      "T_stack": array[25],       # 电堆温度轨迹
                      "SoC": array[25],           # SoC 轨迹
                      "P_cool_elec": array[24],   # 冷却电功率 [W]
                      "Q_cool": array[24],        # 冷却热功率 [W]
                      "P_pump_elec": array[24],   # 泵电功率 [W]
                      "P_fan_elec": array[24],    # 风扇电功率 [W]
                      "P_heat_elec": array[24],   # 加热电功率 [W]
                      "P_agit_elec": array[24],   # 搅拌器电功率 [W]
                      "P_flush_elec": array[24],  # 冲洗泵电功率 [W]
                      "Q_gen": array[24],         # 总产热 [W]
                      "Q_wall": array[24],        # 壁面传热 [W]
                      "Q_shunt": array[24],       # 旁路电流热 [W]
                      "COP": array[24],           # COP
                      "cost_hourly": array[24],   # 每小时冷却电费 [元]
                      "total_cost": float,        # 总冷却电费 [元]
                      "schedule": array[24],      # 冷却计划
                      "heat_schedule": array[24], # 加热计划
                      "heat_hours": int,          # 加热运行时长 [h]
                      # === 新增：能量统计 [kWh] ===
                      "E_internal_kWh": float,    # 内部产热
                      "E_shunt_kWh": float,       # 旁路电流热
                      "E_cool_th_kWh": float,     # 冷却热能
                      "E_heat_th_kWh": float,     # 加热热能
                      "E_wall_kWh": float,        # 壁面传导热
                      "E_cool_elec_kWh": float,   # 冷却电能
                      "E_pump_elec_kWh": float,   # 主泵电能
                      "E_pump_cool_kWh": float,   # 分配给冷却的泵电能
                      "E_pump_not_cool_kWh": float, # 未分配给冷却的泵电能
                      "E_flush_elec_kWh": float,  # 冲洗泵电能
                      "E_heat_elec_kWh": float,   # 加热电能
                      "E_agit_elec_kWh": float,   # 搅拌器电能
                      "E_fan_elec_kWh": float,    # 风扇电能
                      "E_parasitic_total_kWh": float, # 总寄生功耗
                      # === 新增：成本统计 [元] ===
                      "Cost_cool_CNY": float,     # 冷却成本
                      "Cost_fan_CNY": float,      # 风扇成本
                      "Cost_total_CNY": float,    # 总成本
                      # === 新增：运行时长 [h] ===
                      "Cool_h": float,            # 冷却运行时长
                      "Agit_h": float,            # 搅拌器运行时长
                      "Flush_h": float,           # 冲洗运行时长
                      "Fan_h": float,             # 风扇运行时长
                    }
        """
        H = len(price_hourly)
        dt = 3600.0  # 1 小时步长 [秒]

        # 初始化状态
        state = {
            "T_ely":         T_initial,
            "T_stack":       T_initial,
            "SoC":           SoC_initial,
            "Current_State": 0,
            "fan_state":     0,
            "charge_latch":  self.initial_charge_latch(SoC_initial),
        }
        heat_is_running = 0  # 加热状态机初始状态：停止

        # 结果数组
        T_ely_arr = np.zeros(H + 1)
        T_stack_arr = np.zeros(H + 1)
        SoC_arr = np.zeros(H + 1)
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
        COP_arr = np.zeros(H)
        cost_arr = np.zeros(H)
        heat_schedule = np.zeros(H, dtype=int)

        T_ely_arr[0] = T_initial
        T_stack_arr[0] = T_initial
        SoC_arr[0] = SoC_initial

        # === 能量累加变量===
        E_cool_elec_kWh = 0.0
        E_pump_elec_kWh = 0.0
        E_pump_cool_kWh = 0.0
        E_pump_not_cool_kWh = 0.0
        E_flush_elec_kWh = 0.0
        E_heat_elec_kWh = 0.0
        E_agit_elec_kWh = 0.0
        E_fan_elec_kWh = 0.0
        E_charge_grid_kWh = 0.0      # 充电电量
        E_discharge_grid_kWh = 0.0   # 放电电量
        Cost_cool_CNY = 0.0
        Cost_fan_CNY = 0.0

        total_cost = 0.0

        for k in range(H):
            # 窗口第 k 步的钟表时刻（滚动时域；start_hour=0 时等于 k）
            day_h = day_hour_of(start_hour, k)

            # 根据时段判断充放电状态（含 BMS 充电锁存）
            state["Current_State"] = self.determine_charge_discharge_state(
                day_h, state["SoC"], state.get("charge_latch", 0),P_target=(P_target[k] if P_target is not None else None))

            # 构造天气输入
            weather_k = {
                "temperature": float(weather_hourly["temperature"][k]),
                "radiation":   float(weather_hourly["radiation"][k]),
            }

            # 冷却计划
            cool_on = bool(cooling_schedule[k])

            # 加热状态机（如果启用）
            if heat_enabled:
                heat_on = bool(self.determine_heating_state(
                    state["T_ely"], heat_is_running, cool_on))
                heat_is_running = 1 if heat_on else 0
            else:
                heat_on = False

            heat_schedule[k] = 1 if heat_on else 0

            # 执行单步计算
            next_state, metrics = self.step_hourly(
                state, day_h, weather_k, price_hourly[k],
                cool_on=cool_on, heat_on=heat_on,
                P_target=(P_target[k] if P_target is not None else None))

            # 记录结果
            T_ely_arr[k + 1] = next_state["T_ely"]
            T_stack_arr[k + 1] = next_state["T_stack"]
            SoC_arr[k + 1] = next_state["SoC"]
            P_cool_arr[k] = metrics["P_cool_elec"]
            Q_cool_arr[k] = metrics["Q_cool"]
            P_pump_arr[k] = metrics["P_pump_elec"]
            P_fan_arr[k] = metrics["P_fan_elec"]
            P_heat_arr[k] = metrics["P_heat_elec"]
            P_agit_arr[k] = metrics["P_agit_elec"]
            P_flush_arr[k] = metrics["P_flush_elec"]
            Q_gen_arr[k] = metrics["Q_gen"]
            Q_wall_arr[k] = metrics["Q_wall"]
            Q_shunt_arr[k] = metrics["Q_shunt"]
            COP_arr[k] = metrics["COP_now"]
            cost_arr[k] = metrics["cost"]
            total_cost += metrics["cost"]

            # === 能量累加===
            # 电能累加：E += P * dt / 3.6e6  (W × s → kWh)
            E_cool_elec_kWh += metrics["P_cool_elec"] * dt / 3.6e6
            E_pump_elec_kWh += metrics["P_pump_elec"] * dt / 3.6e6
            E_pump_cool_kWh += metrics["pump_cool_share"] * dt / 3.6e6
            E_pump_not_cool_kWh += (metrics["P_pump_elec"] - metrics["pump_cool_share"]) * dt / 3.6e6
            E_flush_elec_kWh += metrics["P_flush_elec"] * dt / 3.6e6
            E_heat_elec_kWh += metrics["P_heat_elec"] * dt / 3.6e6
            E_agit_elec_kWh += metrics["P_agit_elec"] * dt / 3.6e6
            E_fan_elec_kWh += metrics["P_fan_elec"] * dt / 3.6e6

            # 充放电电量累加
            E_charge_grid_kWh += metrics["P_elec_charge"] * dt / 3.6e6
            E_discharge_grid_kWh += metrics["P_elec_disch"] * dt / 3.6e6

            # 成本累加：Cost += P * dt / 3.6e6 * price
            # 冷却成本 = (冷却电功率 + 分配给冷却的泵功率) × 电价
            Cost_cool_CNY += (metrics["P_cool_elec"] + metrics["pump_cool_share"]) * dt / 3.6e6 * price_hourly[k]
            # 风扇成本 = 风扇电功率 × 电价
            Cost_fan_CNY += metrics["P_fan_elec"] * dt / 3.6e6 * price_hourly[k]

            state = next_state

        # === 后处理：复制最后一个时间步的值===
        # for arr in [T_ely_arr, T_stack_arr, SoC_arr, P_cool_arr, Q_cool_arr,
        #             P_pump_arr, P_fan_arr, P_heat_arr, P_agit_arr, P_flush_arr,
        #             Q_gen_arr, Q_wall_arr, Q_shunt_arr, COP_arr]:
        #     arr[-1] = arr[-2]

        # === 能量统计 [kWh]===
        # E = Σ(Q) * dt / 3.6e6  (W × s → kWh)
        E_internal_kWh = float(np.sum(Q_gen_arr) * dt / 3.6e6)
        E_shunt_kWh = float(np.sum(Q_shunt_arr) * dt / 3.6e6)
        E_cool_th_kWh = float(np.sum(Q_cool_arr) * dt / 3.6e6)
        E_heat_th_kWh = float(np.sum(P_heat_arr) * dt / 3.6e6)  # P_heat = Q_heat / COP_heat, COP_heat=1.0
        E_wall_kWh = float(np.sum(Q_wall_arr) * dt / 3.6e6)

        # 总寄生功耗
        E_parasitic_total_kWh = (E_cool_elec_kWh + E_pump_elec_kWh + E_flush_elec_kWh
                                 + E_heat_elec_kWh + E_agit_elec_kWh + E_fan_elec_kWh)

        # 总成本
        Cost_total_CNY = Cost_cool_CNY + Cost_fan_CNY

        # === 往返效率计算===
        # 电堆级 RTE = 放电电量 / 充电电量 × 100%
        RTE_stack = (E_discharge_grid_kWh / E_charge_grid_kWh * 100) if E_charge_grid_kWh > 0 else float("nan")
        # 系统级 RTE = 放电电量 / (充电电量 + 总寄生功耗) × 100%
        RTE_system = (E_discharge_grid_kWh / (E_charge_grid_kWh + E_parasitic_total_kWh) * 100
                      ) if (E_charge_grid_kWh + E_parasitic_total_kWh) > 0 else float("nan")

        # 运行时长 [h]
        Cool_h = float(np.sum(cooling_schedule))  # 冷却计划是 0/1 数组，直接求和
        Agit_h = float(np.sum(P_agit_arr > 0)) * self.agit_duty            # ×10% 占空比
        Flush_h = float(np.sum(P_flush_arr > 0)) * self.flush_duty_cycle   # ×1/30 占空比
        Fan_h = float(np.sum(P_fan_arr > 0))

        result = {
            "T_ely":         T_ely_arr,
            "T_stack":       T_stack_arr,
            "SoC":           SoC_arr,
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
            "COP":           COP_arr,
            "cost_hourly":   cost_arr,
            "total_cost":    total_cost,
            "schedule":      np.array(cooling_schedule),
            "heat_schedule": heat_schedule,
            "heat_hours":    int(np.sum(heat_schedule)),
            # === 新增：能量统计 [kWh] ===
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
            # === 新增：充放电电量 ===
            "E_charge_grid_kWh": E_charge_grid_kWh,
            "E_discharge_grid_kWh": E_discharge_grid_kWh,
            # === 新增：往返效率 ===
            "RTE_stack": RTE_stack,           # 电堆级往返效率
            "RTE_system": RTE_system,         # 系统级往返效率
            # === 新增：成本统计 [元] ===
            "Cost_cool_CNY": Cost_cool_CNY,
            "Cost_fan_CNY": Cost_fan_CNY,
            "Cost_total_CNY": Cost_total_CNY,
            # === 新增：运行时长 [h] ===
            "Cool_h": Cool_h,
            "Agit_h": Agit_h,
            "Flush_h": Flush_h,
            "Fan_h": Fan_h,
        }

        return result


