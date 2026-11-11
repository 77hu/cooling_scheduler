# ============================================================================
# 模块 1：配置与参数管理


# ============================================================================
import datetime
import numpy as np

CN_TZ = datetime.timezone(datetime.timedelta(hours=8))
# ============================================================================
# 保温方案常量
# ============================================================================

INSULATION_FULL = "full"            # 完全保温：所有壁面均有保温层
INSULATION_PARTIAL = "partial"      # 部分保温：一侧长边为裸露钢板

# 集装箱几何参数
A_TOTAL_ENVELOPE = 60.0             # [m²] 集装箱总外壳面积（屋顶 + 四面墙）
A_BARE_LONG_SIDE = 15.7             # [m²] 20 英尺集装箱一侧长边面积
U_INSULATED = 0.5                   # [W/m²K] 保温外壳的有效 U 值（传热系数）
U_BARE_STEEL = 6.5                  # [W/m²K] 裸露钢板的有效 U 值


def get_wall_U(insulation_option):
    """
    根据选择的保温方案返回集装箱壁面的等效 U 值 [W/m²K]，
    均匀应用于整个外壳面积。

    参数:
        insulation_option: 保温方案，"full" 或 "partial"

    返回:
        等效壁面 U 值 [W/m²K]
    """
    if insulation_option == INSULATION_FULL:
        return U_INSULATED
    elif insulation_option == INSULATION_PARTIAL:
        UA_full = U_INSULATED * A_TOTAL_ENVELOPE
        UA_partial = (U_INSULATED * (A_TOTAL_ENVELOPE - A_BARE_LONG_SIDE)
                      + U_BARE_STEEL * A_BARE_LONG_SIDE)
        return UA_partial / A_TOTAL_ENVELOPE
    else:
        raise ValueError(f"未知的保温方案: {insulation_option}")


# ============================================================================
# 数据库连接配置
# ============================================================================

DB_CONFIG = {
    "timezone":    "Asia/Shanghai",
    "ts_host":     "47.107.118.226",
    "ts_port":     5432,
    "ts_user":     "FrankS",
    "ts_password": "Yuanji!!",
    "ts_dbname":   "energy",
    "measurement": "electricity_price_v3",
}

# ============================================================================
# 地理位置配置
# ============================================================================

LOCATION_CONFIG = {
    "location": "Shenzhen_HETAO",       # ✅ 被 data_fetcher / save_database 使用（位置标签）
    "latitude": 22.504639,               # ✅ 被 data_fetcher 使用（天气API坐标）
    "longitude": 114.06105,              # ✅ 被 data_fetcher 使用（天气API坐标）
    "tilt": 2.54,                        # ❌ 未使用 — 光伏倾角，当前无 PV 自消费计算
    "azimuth": 0,                        # ❌ 未使用 — 光伏朝向角
    "pv_kwp": 35.0,                      # ❌ 未使用 — 光伏装机容量 [kWp]
    "rotor_height_m": 2.0,               # ❌ 未使用 — 风机轮毂高度
}

# ============================================================================
# 天气 API 配置
# ============================================================================

WEATHER_API_CONFIG = {
    "timezone": "Asia/Shanghai",
    "models": "best_match",
    "hourly_vars": [
        "temperature_2m",
        "shortwave_radiation",
        "direct_radiation",
        "diffuse_radiation",
        "cloud_cover",
        "relative_humidity_2m",
        "wind_speed_10m",
        "wind_direction_10m",
        "wind_gusts_10m",
        "apparent_temperature",
        "dew_point_2m",
        "direct_normal_irradiance",
        "global_tilted_irradiance",
        "precipitation",
        "weather_code",
    ],
    "forecast_days": 1,                  # ✅ 被 data_fetcher 使用（预报天数）
    "poll_interval_seconds": 3600,       # ❌ 未使用 — 原设计用于轮询天气更新，当前未实现
}


# ============================================================================
# 系统总配置类
# ============================================================================

class SystemConfig:
    """
    集装箱式液流电池储能系统总配置。
    

    用法:
        config = SystemConfig()
        # 访问参数:
        config.stack["P_nom"]           # → 45000
        config.phys["C_th"]             # → 60480000.0
        config.price["windows"]         # → 电价时段表 ndarray
    """

    # DB_CONFIG 中的键，允许通过 config['key'] 下标访问
    _DB_KEYS = ("ts_host", "ts_port", "ts_dbname", "ts_user", "ts_password",
                "measurement", "timezone")

    def __getitem__(self, key):
        """支持 config['ts_host'] 等下标访问，委托到 DB_CONFIG"""
        if key in self._DB_KEYS:
            return DB_CONFIG[key]
        raise KeyError(f"SystemConfig 不支持的键: {key}")

    def __init__(self, insulation_option=INSULATION_FULL,
                 tech_id=1, ekes_scale=1.0, ekes_power_model="COP",
                 days=14, dt=60, vent_active=True,
                 price_list=None):
        """
        初始化系统配置。

        参数:
            insulation_option: 保温方案 ("full" 或 "partial")
            tech_id:           冷却技术类型 (1=板式换热器, 2=EKES)
            ekes_scale:        EKES 容量缩放系数
            ekes_power_model:  EKES 功耗模型 ("COP" 或其他)
            days:              仿真天数                               ❌ 未使用（thermal_model 固定 24h）
            dt:                时间步长 [秒]                          ❌ 未使用（thermal_model 硬编码 3600s）
            vent_active:       是否启用通风                            ✅ 被使用
            price_list:        电价列表 [valley, peak]                 ✅ 被使用
        """
        self.insulation_option = insulation_option
        self.tech_id = tech_id
        self.ekes_scale = ekes_scale
        self.ekes_power_model = ekes_power_model
        self.days = days
        self.dt = dt
        self.vent_active = vent_active

        # 如果未指定电价，使用默认深圳电价
        if price_list is None:
            price_list = [0.23, 1.11]     # [谷时, 峰时]

        # ================================================================
        # 构建各参数组
        # ================================================================

        self._build_insulation()
        self._build_solar()
        self._build_phys()
        self._build_container()
        self._build_stack()
        self._build_pump()
        self._build_hex()
        self._build_cool()
        self._build_heat()
        self._build_agitator()
        self._build_stack_thermal()
        self._build_flush_cfg()
        self._build_vent()
        self._build_price(price_list)
        self._build_optimizer()

        # ================================================================
        # 计算派生量（在子参数组构建完成后计算）
        # ================================================================
        self._compute_derived()

    # ================================================================
    # 各参数组构建方法
    # ================================================================

    def _build_insulation(self):
        """保温方案参数"""
        self.insulation = {
            "option":             self.insulation_option,
            "A_total_envelope":   A_TOTAL_ENVELOPE,       # [m²]
            "A_bare_long_side":   A_BARE_LONG_SIDE,       # [m²]
            "U_insulated":        U_INSULATED,            # [W/m²K]
            "U_bare_steel":       U_BARE_STEEL,           # [W/m²K]
        }
        # 计算等效壁面 U 值
        self.insulation["U_wall_equiv"] = get_wall_U(self.insulation_option)

    def _build_solar(self):
        """
        太阳辐射参数
        说明：thermal_model.py 使用 weather API 的真实逐时辐照度 weather["radiation"]，
              不使用 I_max。I_max 仅在 VRFB.py 的正弦辐射模型中使用。
        """
        self.solar = {
            "I_max":    850.0,    # [W/m²] ❌ 未使用 — 峰值辐照度，
            "alpha":    0.6,      # 外壳太阳能吸收系数     ✅ 被 thermal_model.py 使用
            "h_ext":    35.0,     # [W/m²K] 对流换热系数  ✅ 被 thermal_model.py 使用
            "sunrise":  5.75,     # [小时] 日出时间        ✅ 被 thermal_model.py 使用
            "sunset":   19.17,    # [小时] 日落时间        ✅ 被 thermal_model.py 使用
        }
        # 派生：日照时长 ✅ 被 thermal_model.py 使用
        self.solar["day_len_h"] = self.solar["sunset"] - self.solar["sunrise"]

    def _build_phys(self):
        """电解液物理参数"""
        self.phys = {
            "V_total":  14000.0,  # [L] 电解液总体积
            "rho":      1.35,     # [kg/L] 电堆内滞留电解液体积
            "cp":       3200.0,   # [J/(kg·K)] 比热容
        }
        # 派生：总热容=总体积*密度*比热容
        self.phys["C_th"] = (self.phys["V_total"]
                             * self.phys["rho"]
                             * self.phys["cp"])        # [J/K]

    def _build_container(self):
        """集装箱参数"""
        self.container = {
            "A_tank_surface":  40.0,                    # [m²] 储罐暴露表面积
            "U_tank_inner":    1.5,                     # [W/m²K] 储罐与室内空气传热系数
            "A_wall_inner":    A_TOTAL_ENVELOPE,        # [m²] 集装箱内壁总面积
        }

    def _build_stack(self):
        """电堆参数"""
        self.stack = {
            "active":          True,                   # ❌ 未使用 — 充放电启用标志，
                                                       #    thermal_model.py 按固定时段判断，不检查此标志
            "P_nom":           42000.0,                # [W] 额定电功率 = 45 kW 更改为42KW    ✅
            "Cap_nom":         270000.0,               # [Wh] 额定电容量 = 270 kWh ✅
            "SoC_min":         0.10,                   # 荷电状态下限 [0-1]         ✅ 被 thermal_model.py 使用
            "SoC_max":         0.90,                   # 荷电状态上限 [0-1]         ✅ 被 thermal_model.py 使用
            "SoC_start":       0.50,                   # ❌ 未使用 — 初始 SoC，
                                                       #    由函数参数 SoC_initial 传入
            "start_charge_h":  10.0,                   # [小时] 每日充电开始时间    ✅
            "start_disch_h":   18.0,                   # [小时] 每日放电开始时间    ✅
            "R_ref":           0.015,                  # [Ω] 25°C 时内阻           ✅
            "Ea_R":            2500.0,                 # 电阻阿伦尼乌斯活化能       ✅
            "U_OCV_ref":       55.0,                   # [V] 基准开路电压           ✅
            "dS_coeff":        0.05,                   # 可逆熵热系数               ✅
            "P_shunt_W":       450.0,                  # [W] 100% SoC 时旁路电流损耗 ✅
        }

    def _build_pump(self):
        """主电解液泵参数"""
        self.pump = {
            "ratio_rated":  0.049,           # 额定流量寄生功率比例（4.9% × P_nom）
            "ratio_idle":   0.007,           # 怠速冲洗寄生功率比例
            "Ea_visc":      1800.0,          # 黏度阿伦尼乌斯活化能
        }
        # 派生：额定泵功率和怠速泵功率
        #额定泵功率= 额定电功率*额定流量寄生功率比例
        self.pump["P_rated"] = self.stack["P_nom"] * self.pump["ratio_rated"]   # [W]
        #怠速泵功率= 额定电功率*怠速冲洗寄生功率比例
        self.pump["P_idle"]  = self.stack["P_nom"] * self.pump["ratio_idle"]    # [W]

    def _build_hex(self):
        """换热器参数"""
        self.hex = {
            "active":          True,            # ❌ 未使用 — 冷却启用标志，
                                                #    由优化器决策变量 cool_on 控制，不检查此标志
            # --- 板式换热器（间接冷却，tech_id=1）---
            "plate_UA":        544.0,           # [W/K] 板式换热器 UA 值        ✅
            "plate_T_in":      7.0,             # [°C] 冷水机组入口温度          ✅
            "plate_P_max":     14000.0,         # [W] 冷水机组热功率上限         ✅
            "plate_COP":       2.9,             # 基础性能系数 COP              ✅  
            "LMTD_corr":       0.9,             # 对数平均温差修正系数          ✅
            "COP_deg":         0.08,            # COP 温度衰减因子              ✅
            # --- EKES 管束直接冷却（tech_id=2）---
            # EKES 参考值（通过 ekes_scale 缩放）:
            #   EKES60:  Q=6000W,  P=1300W, COP=4.61 (scale=0.40)
            #   EKES100: Q=10000W, P=2200W, COP=4.55 (scale=0.67)
            #   EKES150: Q=15000W, P=4100W, COP=3.65 (scale=1.00)
            "ekes_Q_nom_ref":    15000.0,       # [W] EKES 基准热功率   ✅
            "ekes_P_elec_ref":   4100.0,        # [W] EKES 基准电功率   ✅
            "ekes_COP_ref":      3.65,          # EKES 基准 COP         ✅
        }
        # 派生：EKES 实际参数（根据缩放系数调整）✅ 被 thermal_model.py 使用
        self.hex["ekes_Q_nom"]      = self.hex["ekes_Q_nom_ref"] * self.ekes_scale
        self.hex["ekes_P_elec_nom"] = self.hex["ekes_P_elec_ref"] * self.ekes_scale
        self.hex["ekes_COP_nom"]    = self.hex["ekes_COP_ref"]

    def _build_cool(self):
        """冷却控制器参数"""
        self.cool = {
            "T_A_on":     35.0,       # [°C] 尖峰电价冷却启动阈值
            "T_A_off":    32.0,       # [°C] 尖峰电价冷却停止阈值
            "T_B_on":     28.0,       # [°C] 低谷电价冷却启动阈值
            "T_B_off":    27.0,       # [°C] 低谷电价冷却停止阈值
            "min_on_s":   300.0,      # [秒] 最小运行时间（防止频繁启闭）
            "min_off_s":  180.0,      # [秒] 最小停机时间
        }

    def _build_heat(self):
        """加热器控制器参数"""
        self.heat = {
            "active":     True,        # 是否启用加热
            "P_heat_W":   5000.0,      # [W] 电加热功率
            "COP_heat":   1.0,         # 加热效率（电阻式加热器通常为 1.0）
            "T_on":       12.0,        # [°C] 加热启动阈值
            "T_off":      15.0,        # [°C] 加热停止阈值
        }

    def _build_agitator(self):
        """电解液搅拌参数"""
        self.agitator = {
            "active":    True,         # 是否启用搅拌器
            "P_agit_W":  300.0,        # [W] 搅拌器电功率
            "period_s":  600.0,        # [秒] PWM 周期
            "pulse_s":   60.0,         # [秒] PWM 脉宽（10% 占空比）
        }
        # 派生：占空比=PWM 脉宽/ PWM 周期
        self.agitator["duty_cycle"] = (self.agitator["pulse_s"]
                                       / self.agitator["period_s"])

    def _build_stack_thermal(self):
        """电堆热质量参数"""
        self.stack_thermal = {
            "m_metal_kg":       150.0,                # [kg] 电堆金属质量
            "cp_metal":         500.0,                # [J/(kg·K)] 金属比热容
            "m_plastic_kg":     350.0,                # [kg] 电堆塑料质量
            "cp_plastic":       1200.0,               # [J/(kg·K)] 塑料比热容
            "V_ely_L":          150.0,                # [L] 电堆内滞留电解液体积
            "dim_m":            [0.877, 0.589, 0.820],# [m] 电堆尺寸 [长, 宽, 高]
            "U_eff_stack":      3.5,                  # [W/m²K] 电堆外壳有效传热系数
            "T_flush_trigger":  10.0,                 # [°C] 触发保温冲洗的温度下降阈值
        }
        # 派生：电堆几何和热参数
        dim = self.stack_thermal["dim_m"]
        #电堆外壳外表面积 [m²]=2（长*宽+长*高+宽*高）
        self.stack_thermal["A_stack"] = 2.0 * (dim[0]*dim[1]
                                                + dim[0]*dim[2]
                                                + dim[1]*dim[2])  # [m²]
        #电堆外壳热导 [W/K]=电堆外壳有效传热系数*电堆外壳外表面积
        self.stack_thermal["UA_stack"] = (self.stack_thermal["U_eff_stack"]
                                          * self.stack_thermal["A_stack"])  # [W/K]
        # 电堆热容各分量=电堆金属质量*金属比热容
        self.stack_thermal["C_th_metal"] = (self.stack_thermal["m_metal_kg"]
                                            * self.stack_thermal["cp_metal"])
        # 电堆塑料件的热容=电堆塑料质量*塑料比热容
        self.stack_thermal["C_th_plastic"] = (self.stack_thermal["m_plastic_kg"]
                                              * self.stack_thermal["cp_plastic"])
        # 电堆内部滞留电解液的热容=电堆内滞留电解液体积*电堆内滞留电解液体积*比热容
        self.stack_thermal["C_th_ely_stk"] = (self.stack_thermal["V_ely_L"]
                                              * self.phys["rho"]
                                              * self.phys["cp"])
        #电堆整体总热容=电堆热容各分量+电堆塑料件的热容+电堆内部滞留电解液的热容
        self.stack_thermal["C_th_stack"] = (self.stack_thermal["C_th_metal"]
                                            + self.stack_thermal["C_th_plastic"]
                                            + self.stack_thermal["C_th_ely_stk"])

    def _build_flush_cfg(self):
        """电堆冲洗配置"""
        self.flush_cfg = {
            "duration_s":  120.0,       # [秒] 冲洗泵运行时长
        }

    def _build_vent(self):
        """集装箱通风参数"""
        self.vent = {
            "Flow_m3h":         2000.0,      # [m³/h] 设计体积流量
            "P_ref":            150.0,       # [W] 风机参考功率
            "Flow_ref":         2000.0,      # [m³/h] 风机参考流量
            "dT_trigger_on":    3.0,         # [K] 通风启动温差
            "dT_trigger_off":   1.0,         # [K] 通风停止温差
        }
        # 派生：风机实际功率（风机定律：功率与流量的三次方成正比）
        self.vent["P_fan"] = (self.vent["P_ref"]
                              * (self.vent["Flow_m3h"]
                                 / self.vent["Flow_ref"]) ** 3)
        # 派生：通风残差因子（反映通风对散热的实际贡献）
        self.vent["residual_factor"] = (220.0/ (220.0 + self.vent["Flow_m3h"]))

    def _build_price(self, price_list):
        """
        分时电价参数（TOU）
        深圳大型工业/商业（10 kV）费率
        谷:平:峰 = 0.3142:1:1.5445，尖峰 = 峰 × 1.25
        """
        
        valley   = price_list[0]
        peak     = price_list[1]
        flat     = peak*0.47+valley*0.53
        critical = peak * 1.25

        self.price = {
            "valley":   valley,         # [元/kWh] 谷时电价
            "flat":     flat,           # [元/kWh] 平时电价
            "peak":     peak,           # [元/kWh] 峰时电价
            "critical": critical,       # [元/kWh] 尖峰电价
            "default":  flat,           # [元/kWh] 默认电价

            # 电价等级映射
            "levels": {
                1: valley,              # 等级 1：谷时
                2: flat,                # 等级 2：平时
                3: peak,                # 等级 3：峰时
                4: critical,            # 等级 4：尖峰
            },

            # 电价时段表：[起始小时, 结束小时, 等级]
            # 等级 1=谷, 2=平, 3=峰, 4=尖峰
            "windows": np.array([
                [0.0,   8.0,  1],      # 00:00-08:00 谷时
                [8.0,  10.0,  2],      # 08:00-10:00 平时
                [10.0, 12.0,  3],      # 10:00-12:00 峰时
                [12.0, 14.0,  2],      # 12:00-14:00 平时
                [14.0, 19.0,  3],      # 14:00-19:00 峰时
                [19.0, 24.0,  2],      # 19:00-24:00 平时
            ]),
        }


    def _build_optimizer(self):
        """优化器参数"""
        self.optimizer = {
            "T_min":            20.0,     # [°C] 状态空间下限          ✅
            "T_max":            40.0,     # [°C] 状态空间上限          ✅
            "T_step":           0.25,      # [°C] 温度离散步长          ✅
            "T_target_low":     25.0,     # [°C] 温度舒适下限          ✅
            "T_target_high":    35.0,     # [°C] 温度舒适上限          ✅
            "horizon_h":        24,       # [小时] 优化时域            ✅
            "peak_price_threshold": 1.0,  # [元/kWh] ❌ 未使用 — 区分峰/谷的电价阈值，
                                          #    baseline.py 使用 TOU 时段表判断峰谷，非电价数值
            # "run_penalty":      2.5,      # [元/h] 运行时长惩罚系数    ✅
            "startup_penalty":3.0,            #[元/次] 冷却启停惩罚（状态翻转一次罚一次）✅
            "T_dyn_base":38.0,
            "T_dyn_slope":4.0,
        }

    # ================================================================
    # 派生量计算
    # ================================================================

    def _compute_derived(self):
        """
        计算跨参数组的派生量。
        在所有子参数组构建完成后调用。
        """
        # 集装箱热阻网络参数
        #储罐(电解液）→箱内空气的热导=储罐与室内空气传热系数*储罐暴露表面积
        self.UA_tank_inner = (self.container["U_tank_inner"]
                              * self.container["A_tank_surface"])     # [W/K]
        #箱壁(外界)→箱内空气的热导 =装箱壁面的等效传热系数*集装箱内壁总面积
        self.UA_wall_inner = (self.insulation["U_wall_equiv"]
                              * self.container["A_wall_inner"])       # [W/K]
        self.UA_total = self.UA_tank_inner + self.UA_wall_inner       # [W/K]

    # ================================================================
    # 便捷方法
    # ================================================================

    def get_price_and_tariff(self, day_h):
        """
        根据当前时刻获取电价和电价状态。
        复刻原始 get_price_and_tariff() 函数。

        参数:
            day_h: 当前时刻（一天中的小时数，0-24）

        返回:
            (当前电价, 电价状态字符串)
            电价状态："peak"（峰时）或 "off_peak"（谷时）
        """
        levels = self.price["levels"]

        for row in self.price["windows"]:
            h0, h1, lid = row[0], row[1], int(row[2])
            if h0 <= day_h < h1:
                p = levels.get(lid, self.price["default"])
                ts = "peak" if lid == 3 else "off_peak"
                return p, ts

        return self.price["default"], "off_peak"

    def get_tariff_label(self, day_h):
        """
        根据当前时刻获取电价等级标签（中文）。

        参数:
            day_h: 当前时刻（一天中的小时数，0-24）

        返回:
            电价等级标签字符串："谷"、"平"、"峰"、"尖峰"
        """
        label_map = {1: "谷", 2: "平", 3: "峰", 4: "尖峰"}

        for row in self.price["windows"]:
            h0, h1, lid = row[0], row[1], int(row[2])
            if h0 <= day_h < h1:
                return label_map.get(lid, "平")

        return "平"

    def get_tariff_label_by_price(self, price):
        """
        根据电价数值获取最接近的电价等级标签。
        用于无法获取小时信息的场景。

        参数:
            price: 电价数值 [元/kWh]

        返回:
            电价等级标签字符串："谷"、"平"、"峰"、"尖峰"
        """
        # 计算与各等级的距离，返回最近的
        levels = self.price["levels"]
        label_map = {1: "谷", 2: "平", 3: "峰", 4: "尖峰"}

        min_dist = float("inf")
        best_label = "平"
        for lid, p in levels.items():
            dist = abs(price - p)
            if dist < min_dist:
                min_dist = dist
                best_label = label_map.get(lid, "平")

        return best_label

    def print_summary(self):
        """打印配置摘要，用于调试验证"""
        print("\n" + "=" * 60)
        print("  系统配置摘要")
        print("=" * 60)

        print(f"\n--- 保温方案 ---")
        print(f"  方案: {self.insulation['option']}")
        print(f"  等效 U 值: {self.insulation['U_wall_equiv']:.3f} W/m²K")

        print(f"\n--- 电解液物理参数 ---")
        print(f"  总体积: {self.phys['V_total']:.0f} L")
        print(f"  密度: {self.phys['rho']:.2f} kg/L")
        print(f"  比热容: {self.phys['cp']:.0f} J/(kg·K)")
        print(f"  总热容 C_th: {self.phys['C_th']:.0f} J/K")

        print(f"\n--- 集装箱热阻网络 ---")
        print(f"  UA_tank_inner: {self.UA_tank_inner:.1f} W/K")
        print(f"  UA_wall_inner: {self.UA_wall_inner:.1f} W/K")
        print(f"  UA_total: {self.UA_total:.1f} W/K")

        print(f"\n--- 电堆 ---")
        print(f"  额定功率: {self.stack['P_nom']:.0f} W")
        print(f"  额定容量: {self.stack['Cap_nom']:.0f} Wh")
        print(f"  SoC 范围: {self.stack['SoC_min']:.0%} ~ "
              f"{self.stack['SoC_max']:.0%}")
        print(f"  基准内阻: {self.stack['R_ref']:.3f} Ω")

        print(f"\n--- 泵 ---")
        print(f"  额定功率: {self.pump['P_rated']:.1f} W")
        print(f"  怠速功率: {self.pump['P_idle']:.1f} W")
        print(f"  黏度活化能: {self.pump['Ea_visc']:.0f}")

        print(f"\n--- 换热器 (tech_id={self.tech_id}) ---")
        if self.tech_id == 1:
            print(f"  类型: 板式换热器（间接冷却）")
            print(f"  UA: {self.hex['plate_UA']:.1f} W/K")
            print(f"  冷源入口温度: {self.hex['plate_T_in']:.1f} °C")
            print(f"  最大功率: {self.hex['plate_P_max']:.0f} W")
            print(f"  基准 COP: {self.hex['plate_COP']:.1f}")
            print(f"  LMTD 修正: {self.hex['LMTD_corr']:.1f}")
            print(f"  COP 衰减因子: {self.hex['COP_deg']:.2f}")
        else:
            print(f"  类型: EKES（直接冷却）")
            print(f"  额定热功率: {self.hex['ekes_Q_nom']:.0f} W")
            print(f"  额定电功率: {self.hex['ekes_P_elec_nom']:.0f} W")
            print(f"  基准 COP: {self.hex['ekes_COP_nom']:.2f}")

        print(f"\n--- 冷却控制 ---")
        print(f"  尖峰电价启停: {self.cool['T_A_on']:.0f}/{self.cool['T_A_off']:.0f} °C")
        print(f"  谷时电价启停: {self.cool['T_B_on']:.0f}/{self.cool['T_B_off']:.0f} °C")
        print(f"  最小运行时间: {self.cool['min_on_s']:.0f} s")
        print(f"  最小停机时间: {self.cool['min_off_s']:.0f} s")

        print(f"\n--- 加热 ---")
        print(f"  加热功率: {self.heat['P_heat_W']:.0f} W")
        print(f"  启停阈值: {self.heat['T_on']:.0f}/{self.heat['T_off']:.0f} °C")

        print(f"\n--- 搅拌器 ---")
        print(f"  功率: {self.agitator['P_agit_W']:.0f} W")
        print(f"  占空比: {self.agitator['duty_cycle']:.0%}")

        print(f"\n--- 电堆热质量 ---")
        print(f"  表面积: {self.stack_thermal['A_stack']:.3f} m²")
        print(f"  UA_stack: {self.stack_thermal['UA_stack']:.1f} W/K")
        print(f"  总热容: {self.stack_thermal['C_th_stack']:.0f} J/K")

        print(f"\n--- 通风 ---")
        print(f"  风机功率: {self.vent['P_fan']:.1f} W")
        print(f"  残差因子: {self.vent['residual_factor']:.4f}")
        print(f"  启停温差: {self.vent['dT_trigger_on']:.0f}/{self.vent['dT_trigger_off']:.0f} K")

        print(f"\n--- 分时电价 ---")
        print(f"  谷时: {self.price['valley']:.4f} 元/kWh")
        print(f"  平时: {self.price['flat']:.4f} 元/kWh")
        print(f"  峰时: {self.price['peak']:.4f} 元/kWh")
        print(f"  尖峰: {self.price['critical']:.4f} 元/kWh")

        print(f"\n--- 仿真参数 ---")
        print(f"  天数: {self.sim['days']}")
        print(f"  步长: {self.sim['dt']} s")
        print(f"  总步数: {self.sim['steps']}")

        print(f"\n--- 优化器参数 ---")
        print(f"  状态空间: {self.optimizer['T_min']}~{self.optimizer['T_max']} °C")
        print(f"  离散步长: {self.optimizer['T_step']} °C")
        print(f"  舒适区间: {self.optimizer['T_target_low']}~{self.optimizer['T_target_high']} °C")
        print(f"  优化时域: {self.optimizer['horizon_h']} h")

        print("\n" + "=" * 60)


# ============================================================================
# 未使用参数汇总（供维护参考）
# ============================================================================
# 以下参数在 config.py 中定义，但在 cooling_scheduler 优化流程中未被任何模块读取：
#
# 1. LOCATION_CONFIG:
#    - tilt, azimuth, pv_kwp, rotor_height_m    （光伏/风机参数，无 PV 自消费计算）
#
# 2. WEATHER_API_CONFIG:
#    - poll_interval_seconds                     （轮询间隔，未实现轮询机制）
#
# 3. SystemConfig.env:
#    - T_max, T_min                              （环境温度默认值，被 weather API 覆盖）
#
# 4. SystemConfig.solar:
#    - I_max                                     （峰值辐照度，thermal_model 用真实 weather 数据）
#
# 5. SystemConfig.stack:
#    - active                                    （充放电启用标志，thermal_model 按固定时段判断）
#    - SoC_start                                 （初始 SoC，由函数参数 SoC_initial 传入）
#
# 6. SystemConfig.hex:
#    - active                                    （冷却启用标志，由优化器决策变量 cool_on 控制）
#
# 7. SystemConfig.sim:
#    - days, dt, steps                           （仿真时长/步长，thermal_model 硬编码 1h 步长，
#                                                 仿真长度由输入数据长度决定）
#
# 8. SystemConfig.optimizer:
#    - peak_price_threshold                      （峰谷电价阈值，baseline 用 TOU 时段表判断）
#
# 9. SystemConfig 构造函数参数:
#    - days, dt                                  （同上，传入后存入 sim 但不被使用）
#

# ============================================================================
