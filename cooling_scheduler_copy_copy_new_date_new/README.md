# VRFB 冷却计划滚动优化器（cooling_scheduler_copy_copy_new_date_new）

> 面向集装箱式全钒液流电池（VRFB，42 kW / 270 kWh 标称）储能系统的冷却调度控制器。
> 基于详细热力学模型 + 动态规划（DP）+ 小时级滚动时域重优化（MPC 式），
> 在保证电解液温度安全的前提下，结合真实天气预报、真实分时电价、**数据库实测
> 充放电功率 / SoC / 电解液温度**，求解经济性最优的冷却机组逐时开/关计划。

> **文档版本**：2026-09-09 · 基于当前代码逐行核对 + 现场数据库实测验证更新。
> 本版新增：第十章（真实数据接入层）、第十一章（实时运行层重构）、第十五章
> （现场实测与模型参数对照）、第十六章（已知问题路线图）；修正了旧版与代码
> 不符的描述（如 §8.2 P_target 生效范围）。

---

## 目录

- [一、项目定位与演化谱系](#一项目定位与演化谱系)
- [二、当前文件清单](#二当前文件清单)
- [三、系统架构](#三系统架构)
- [四、核心设计理念：滚动时域](#四核心设计理念滚动时域)
- [五、物理热力学模型（thermal_model.py）](#五物理热力学模型thermal_modelpy)
- [六、DP 优化器（optimizer.py）](#六dp-优化器optimizerpy)
- [七、数据获取与容错（data_fetcher.py）](#七数据获取与容错data_fetcherpy)
- [八、滚动调度核心（rolling_runner.py）](#八滚动调度核心rolling_runnerpy)
- [九、外部调度接口 P_target 与符号约定](#九外部调度接口-p_target-与符号约定)
- [十、真实数据接入层（save_database.py）★](#十真实数据接入层save_databasepy)
- [十一、实时运行层（live_runner.py）★](#十一实时运行层live_runnerpy)
- [十二、命令行入口（main_rolling.py）](#十二命令行入口main_rollingpy)
- [十三、可视化（live_visualizer.py）](#十三可视化live_visualizerpy)
- [十四、全局配置参数（config.py）](#十四全局配置参数configpy)
- [十五、现场实测数据与模型参数对照 ★](#十五现场实测数据与模型参数对照)
- [十六、已知问题与改进路线图](#十六已知问题与改进路线图)
- [十七、验证记录](#十七验证记录)
- [十八、附录：常用 SQL 与工具脚本](#十八附录常用-sql-与工具脚本)

---

## 一、项目定位与演化谱系

系统要解决的问题：

> 给定未来 24 小时的天气预报、分时电价、以及电池实际的充放电功率，
> 如何安排冷却机组的逐时开/关，使电解液温度全程维持在安全区间内，
> 同时让冷却产生的电费尽可能低？

```
VRFB.py v5.2            原型：60s×14天时序仿真，TOU双阈值滞回规则，合成天气
   ↓ 重构
cooling_scheduler v1.0  1h×24h 日前DP优化，Open-Meteo真天气 + PostgreSQL真电价
   ↓ 升级
cooling_scheduler_copy  + 小时级滚动时域(MPC) + 闭环验证平台 + --live实时运行
   ↓ 裁剪 + 真实数据改造（★当前目录）
本项目                   接入现场 mqtt 数据库：实测温度/功率/SoC 三锚点 +
                        符号约定翻译 + 指定日期回测 + 电价采集子进程联动
```

---

## 二、当前文件清单

本目录是裁剪版（对照/验证类模块 baseline.py、validator.py、comparator.py、
closed_loop_validation.py、output_formatter.py、visualizer.py、main.py 保留在
旧项目 `cooling_scheduler/` 中，本目录不含）。

| 文件 | 行数 | 职责 |
|---|---|---|
| `config.py` | 659 | SystemConfig：17 组参数 + 派生量 + TOU 时段表 + DB_CONFIG |
| `data_fetcher.py` | 565 | Open-Meteo 天气（历史/预报双 API）+ PG 电价（三级回退）+ 缓存 |
| `thermal_model.py` | 1059 | 1h 步长热-电化学引擎：step_hourly / simulate_24h / 充放电状态机 / charge_latch / day_hour_of |
| `optimizer.py` | 535 | DP：81温度×21SoC×2动作×2锁存，可达集剪枝，~25s/轮 |
| `rolling_runner.py` | 244 | run_once 七步：重解+冻结承诺+安全覆写+锁存传递；MeasuredState / RunContext |
| `live_runner.py` | ~290 | ★ 实时/回测主循环：三锚点同步取数 + 逐轮滚动结算 + get_dispatch_target |
| `live_visualizer.py` | 351 | --live 模式 5 张独立图（温度/计划甘特/电价/电费/SoC） |
| `save_database.py` | ~440 | ★ 3 个 DB 类：DB_RollingResult（结果入库）/ Real_time_db（实测取数）/ Weather_data（天气入库） |
| `main_rolling.py` | ~220 | 命令行入口：--live 四模式 + 电价采集子进程启动器 |
| `VRFB.py` | 776 | 历史原型（独立脚本，不参与主链路） |
| `test_real_power.py` | 111 | 功率测点取数测试脚本（开发期工具） |
| 父目录 `query_power_range.py` | ~70 | 指定时间范围原始数据导出工具（开发期工具） |

---

## 三、系统架构

```
┌─ 数据层 ────────────────────────────────────────────────────────────┐
│ data_fetcher.py   Open-Meteo(过去→archive历史API/未来→forecast API) │
│                   PostgreSQL energy库 electricity_price_v3(三级回退) │
│ save_database.py  Real_time_db → PostgreSQL mqtt库 BMS表(340列,~7s/条)│
│   ├ get_current_max_temp  9路电解液温度 5min窗口均值取最大 → T_ely   │
│   ├ get_real_power        PCS功率+总电流 → 死区/判向/符号翻译 → P_target│
│   └ get_SoC               SOC_CPU/HMI → 归一化0~1 → SoC锚点          │
│   Weather_data            Open-Meteo历史天气 → energy.weather_data表  │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼  三路实测共用同一 as_of=ready_at 时间锚点
┌─ 物理层 ────────────────────────────────────────────────────────────┐
│ thermal_model.py  3节点热网络 + Nernst OCV + Arrhenius内阻 + 熵热    │
│                   + 法拉第SoC积分 + shunt自放电 + 充放电状态机        │
│                   + BMS充电锁存 + 边界截断 + 冷却设备模型(板式/EKES)  │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─ 决策层 ────────────────────────────────────────────────────────────┐
│ optimizer.py      DP: 前向可达集剪枝 → 后向Bellman递推 → 前向追踪    │
│                   目标 = 冷却电费 + 启停惩罚 + 温度/SoC越限惩罚       │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─ 滚动层 ────────────────────────────────────────────────────────────┐
│ rolling_runner.py run_once七步; 优先级: 安全覆写 > 冻结承诺 > DP原始  │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─ 运行层 ────────────────────────────────────────────────────────────┐
│ live_runner.py    每小时一轮×24: 三锚点取数 → run_once → 承诺首步     │
│                   → _settle_hour结算 → CSV+5图                       │
│ main_rolling.py   CLI入口 + power_price电价采集子进程(非阻塞)        │
└──────────────────────────────────────────────────────────────────────┘
```

模块依赖单向自底向上：config ← thermal_model ← optimizer ← rolling_runner
← live_runner ← main_rolling。每层可独立测试替换。

---

## 四、核心设计理念：滚动时域

日前一次性优化（daily）的根本问题：**预报误差随时间累积，计划定死后无法纠偏**。

本项目采用**滚动时域（Receding Horizon / MPC）**：

- 每小时重新执行完整流程：最新天气 + 最新电价 + **当前实测**温度/SoC/功率
  作为起点 → 重解未来 24h 最优计划；
- 只**执行**第一步（承诺动作），其余 23 步仅参考；
- 下一轮起点用**真实测量值**覆盖（温度/SoC 不靠模型外推），预报误差每小时
  被"清零"一次，不跨小时累积。

代价：每轮完整 DP 求解 ~25 秒（剪枝后），相对 1 小时调度周期可接受。

---

## 五、物理热力学模型（thermal_model.py）

`DetailedThermalModel.step_hourly()` 是全项目唯一物理引擎，DP 每次试探、
滚动每次结算都调用它。单步计算按物理因果顺序：

1. **环境侧**：太阳辐射 → 外壳等效温升 `dT_solar = α×I/h_ext`，与气温叠加成
   Sol-Air 综合温度；通风风扇滞环控制（开 3K / 关 1K 温差），防抖。
2. **箱内空气**：储罐→空气、箱壁→空气两条热导加权代数求解（空气热容忽略，
   减一个状态维度）。
3. **冷却功率**（tech_id 二选一）：
   - 板式换热器（默认）：`Q = min(UA×ΔT×0.9, 14000) W`，UA=544 W/K；
     `COP = max(2.9 − 0.08×ΔT_amb, 1.5)`；待机冷却免主泵
   - EKES 直冷：15kW×scale，≤22℃ 停机；`COP = max(3.65 − 0.05×ΔT_amb, 2.0)`；
     冷却时主泵必须额定
4. **充放电与产热**：
   - 内阻 Arrhenius：`R = 0.015 × exp(2500×(1/T − 1/298.15))` Ω
   - 开路电压类 Nernst：`U_OCV = 55 + 1.5×ln(SoC/(1−SoC))` V
   - 电流反解（P=U×I, U=OCV+I×R）：
     充电 `I = (−U+√(U²+4RP))/(2R)`；放电 `I = (U−√(U²−4RP))/(2R)`
   - **产热不对称**：充电 `Q = I²R − I·T·dS`（熵热吸热）；
     放电 `Q = I²R + I·T·dS`（熵热放热），dS_coeff=0.05
   - 部分负载：有 P_target 时 `P_chg=min(|P|,P_nom)`、`P_dis=min(|P|,P_max×0.95)`；
     无 P_target 退回额定
5. **SoC 法拉第积分**：`ΔSoC = I×55V×1h / 270kWh`（常数 55V 计量近似）
6. **边界截断**：本步撞 SoC_max/SoC_min 时按活跃分数 `f = headroom/ΔSoC`
   缩放产热、泵功率、电量——模拟 BMS 到限即停。
7. **BMS 充电锁存**（charge_latch）：充电撞限置位，窗口内禁止再充（防 shunt
   自放电拖低 SoC 后出现涓流补电脉冲）；离开充电窗口清除。取代旧 eps 容差。
8. **shunt 自放电**：`Q_shunt = 450×SoC W`，对应 SoC 损失计入积分。
9. **温度更新**（唯一状态推进公式）：
   `T_ely' = T_ely + (Q_gen + Q_wall − Q_cool + Q_heat)×dt / C_th`，
   C_th = 60.48 MJ/K（14000L×1.35×3200），集装箱 UA=90 W/K → 时间常数≈7.8天
10. **成本口径**：只计冷却机组电费 + 分配给冷却的泵功耗；充放电泵/搅拌/风扇
    等寄生功耗不进 DP 目标（不受冷却决策影响，混入会干扰搜索方向）。

**充放电状态机** `determine_charge_discharge_state(hour, SoC, latch, P_target)`：
- 有 P_target → `_state_from_power`：`p>0 且 SoC>SoC_min` → 放电(2)；
  `p<0 且未锁存且 SoC<SoC_max` → 充电(1)；否则待机(0)
- 无 P_target → 时间窗口兜底：充电 10:00~17:00，放电 18:00~次日6:00
  （`start_charge_h`/`end_charge_h=17`/`start_disch_h`，止点硬编码）

**`day_hour_of(start_hour, k) = (start_hour+k) % 24`**：滚动窗口第 k 步 →
钟表时刻映射，支持任意起始小时的 24h 窗口（start_hour=0 时与旧版逐位一致）。

---

## 六、DP 优化器（optimizer.py）

### 6.1 状态空间（四维）

```
状态 = (温度 t: 81格 20~40℃/0.25, SoC s: 21格 0~1/0.05,
        上步冷却 p: 2, 充电锁存 l: 2)   → 13,608 状态 × 2 动作 × 24 步
```

- p 维：让"启停惩罚"具备马尔可夫性（不知道上步开关就无法判断翻转）
- l 维：让"本格是否允许充电"不依赖历史路径，后向递推才成立

### 6.2 三段式求解

1. **前向可达集剪枝**：从真实起点正向推 24 步（2锁存×2动作全遍历，宁多勿漏），
   记录可达 (t,s) 集合并外扩 ±1 格容错；全网格 32.6 万次调用 → 剪枝后 ~25s/轮。
2. **后向 Bellman 递推**：h=23→0，每个可达状态试开/关两动作：
   `total = 即时成本 + V[h+1][t'][s'][a][l']`，取小者入 policy 表。
   即时成本 = 冷却电费(metrics.cost) + 温度越限惩罚 + 启停惩罚(3元/次)。
   - 过热惩罚 `10×(T−T_hi)²`（T_hi = min(35, 38−4×SoC) 动态上限）
   - 过冷惩罚 `5×(25−T)²`；权重不对称：过热威胁安全，过冷只浪费电
   - 终值惩罚：温度偏离 + SoC 越限 `20×越(0.15/0.85)²`（最重）
3. **前向追踪**：从初始状态查 policy 表走 24 步，产出计划/温度/SoC/成本轨迹。
   前向用连续 SoC（非网格值），查表时 nearest 量化。

### 6.3 ⚠ P_target 在 DP 内的当前生效范围（与旧版文档不同，以代码为准）

| 阶段 | determine_charge_discharge_state | step_hourly |
|---|---|---|
| 可达集剪枝 | ✅ 传 P_target[hh] | ❌ 未传（额定功率，保守上界，有意为之） |
| 后向递推 | ✅ 传 P_target[h]（L260） | ❌ **该行被注释**（L283，"等待接入真数据打开测试"） |
| 前向追踪 | ✅ 传 P_target[h]（L395） | ❌ **该行被注释**（L409） |

**后果**：状态判断用实测功率（部分负载），但产热/SoC 积分按额定 42kW 计算。
19.65kW 充电时产热高估约 2 倍；且 `_settle_hour` 结算传了 P_target（部分负载），
规划与结算口径不一致。→ 见第十六章 P1-2，真实数据已接入，应打开这两行。

---

## 七、数据获取与容错（data_fetcher.py）

### 7.1 天气 fetch_weather_window()

- **日期感知双 API**：窗口在过去 → Open-Meteo **archive 历史 API**
  （start_date/end_date 按日拉取，支撑指定日期回测）；现在/未来 →
  **forecast API**（forecast_days=2 拉 48 点覆盖跨午夜窗口）
- 按**时间戳**（非裸下标）定位窗口起点，防 API 起始时刻漂移导致错位
- 失败 → 上次成功缓存 `_LAST_GOOD_WEATHER_WINDOW` → 再失败返回 None
  （上游 no_data_hold 保持上轮动作，fail-operational）

### 7.2 电价 resolve_price_pairs() 三级回退

```
① 当日 DB 真实谷/峰价对 → ② "今天"的价对代替 → ③ 默认 0.23/1.11 元
```

按**日历日**回退（电价整天发布），跨午夜窗口两天各自独立回退。
`build_price_hourly()` 用固定时段表把谷/峰价展开成逐时价。

---

## 八、滚动调度核心（rolling_runner.py）

`run_once(run_dt, meas, ctx, P_target=None)` 七步：

| 步骤 | 做什么 | 为什么 |
|---|---|---|
| 1 天气窗口 | 失败→缓存→None(no_data_hold) | API 抖动不停摆 |
| 2 电价拼接 | 每日历日独立三级回退 | 跨午夜窗口隔离单日缺失 |
| 3 DP 重解 | 实测初值+显式锁存+start_hour 钟表映射 | 每轮全新求解 |
| 4 冻结承诺 | 首步与上轮承诺冲突→维持承诺(commit_hold) | 防相邻两轮解抖动导致设备频繁启停 |
| 5 安全覆写 | 实测温度 ≥38℃ 强开 / ≤18℃ 强关 | 硬约束压过一切经济性权衡，用实测不用预报 |
| 6 登记承诺 | 未来 commit_horizon_h(=1) 小时入承诺表+清过期 | 给下轮第4步比对依据 |
| 7 锁存传递 | 模型重推一步算出执行后 charge_latch | 离散状态无法从连续量反推，必须显式跨轮 |

**优先级链：安全覆写 > 冻结承诺 > DP 原始决策**。
返回值含 `raw_first_action`（DP原始）与 `first_action`（实际执行）+ `overrides`
列表，全程可审计。

`MeasuredState`：T_ely / T_stack / SoC / charge_latch / prev_cool_action / fan_state
（T_ely、SoC 为必填，由 live_runner 每轮实测锚定）。

---

## 九、外部调度接口 P_target 与符号约定

### 9.1 设计目的

把"何时充放电、功率多大"从热模型硬编码时段中剥离为外部注入参数，适配任意
上级调度（EMS/人工计划/实测延续）。单位 W，24 长度数组按窗口步对齐。

### 9.2 ★ 符号约定总表（两套约定，方向相反，边界处翻译一次）

| 世界 | 约定 | 证据 |
|---|---|---|
| **数据库** PCS Real-Time Power | **正=充电，负=放电**（电池视角：流入为正） | 2026-09-09 充电会话：SOC_HMI 上升段 224 样本 100% 对应 PCS>0，无一例外 |
| **数据库** Total Current（BMS） | **负=充电，正=放电**（与 PCS 功率反号） | 充电时 −225A / 放电时 +146A |
| **模型** P_target（_state_from_power） | **负=充电，正=放电，0=待机**（电网视角：出力为正） | thermal_model 源码 + get_dispatch_target 计划表（−25000 谷充 / +30000 峰放） |

**翻译点（全项目唯一）**：`get_real_power` 内 `p_target = -pcs_w`。
删除这行取反 → 充电被当放电：熵热符号反转 + 电流解支路错 → 30kW 工况下
产热误差约 20kW（充电真实净吸热 −3.7kW，误判放电算成 +16.6kW），SoC 轨迹
反向，dynamic_T_max 安全边界漂移。

### 9.3 P_window 组装（live_runner.get_dispatch_target）

```python
P_window = 硬编码24点计划表[(start_hour+k)%24]      # 尾部23格：计划值
P_window[0] = p_info['p_target']                    # 第0格：数据库实测覆盖
```

第 0 格是唯一被承诺执行的小时 → 用实测；尾部用计划表，每轮滚动刷新（MPC
容错）。⚠ 当前计划表与现场实际调度不符（表：14-18点放电；现场 9-9 实测
11:00-15:30+ 充电 30kW）→ 见第十六章 P0-3。

---

## 十、真实数据接入层（save_database.py）★

### 10.1 数据库部署

| 库 | 表 | 内容 | 采样 |
|---|---|---|---|
| **mqtt** (47.107.118.226:5432) | `BMS` | 340 列现场遥测/遥信（温度/功率/SoC/压力/流量/状态位） | ~6-8s/条 |
| mqtt | Fan_1/2、Photovoltaic、Statistic、Electric_Meter_* 等 11 张分表 | 风机/光伏/电表等 | — |
| **energy** | `electricity_price_v3` | 逐日谷/峰电价对（power_price.py 爬虫维护） | 日级 |
| energy | `weather_data` | Open-Meteo 历史天气落地（Weather_data 类维护） | 日级 |
| energy | `cooling_rolling_runs` / `cooling_applied_hours` | 本项目调度结果（DB_RollingResult 写入） | 轮/小时级 |

### 10.2 DB_RollingResult（结果入库）

- `cooling_rolling_runs`：每轮一行，**ON CONFLICT DO NOTHING**（历史不可变，
  复盘"当时为什么这么决策"）
- `cooling_applied_hours`：每执行小时一行，**DO UPDATE 只补真值列**（动作保持
  首写，事后补 truth_ambient/truth_price/step_cost 审计）
- 写库失败只告警不抛异常（fail-operational，存储故障不拖垮控制主链路）
- ⚠ 当前 live 模式未调用入库（只写 CSV）→ 第十六章 P1-5

### 10.3 Real_time_db（实测取数，本项目核心新增）

**连接韧性**：`_ensure_conn()` + `_fetchone(sql, params)` 包装——查询遇
`OperationalError`（服务端断连）自动重连重试一次。长时运行必备
（现场数据库已出现过连接超时）。

#### ① get_current_max_temp(as_of, window_minutes=5, fallback=28.0)

- 测点：9 路电解液相关温度（TEMP_COLS）：
  正/负极电解液罐上/下部 ×4、正/负极回水管上/下游 ×4、集装箱顶部 ×1
  （Liquid Temperature 为冷却液温度，也在列，共 10 列，以代码 TEMP_COLS 为准）
- 算法：`[as_of−5min, as_of]` 窗口内每测点 avg → 取 **max** → 锚点 T_ely
  （保守设计：盯最热点，任何一处超温都触发冷却）
- 无数据 → fallback 28.0（并打印告警）；`return_name=True` 返回最热测点名
- 时区：naive as_of 自动补 CN_TZ(+08:00)

#### ② get_real_power(as_of, max_stale_sec=60) → 单 info 字典

查询：`SELECT timestamp, "PCS Real-Time Power", "Total Current" FROM "BMS"
WHERE timestamp <= as_of ORDER BY timestamp DESC LIMIT 1`
——**同一行取多列 = 同一时刻快照**（7 秒内），天然对齐，无需多次查询。

合成逻辑（当前版）：

```
pcs_w = PCS Real-Time Power × 1000        [W]，数据库符号：正=充电
EPS_W = 1000 W   功率死区（待机噪声实测达 −640W，500 会被穿透）
EPS_A = 2 A      电流死区（待机噪声 −0.36~−0.69A）

|pcs_w| < EPS_W          → source=-1 待机, p_target=0
否则 p_target = -pcs_w   → 符号翻译（数据库→模型约定）
     source = 1(放电) if Total Current > EPS_A else 0(充电)
```

返回 info 键：`data_ts`（数据行时刻）/ `stale`（距 as_of > 60s 即数据过期）/
`pcs_kw`（数据库原始值，对账用）/ `p_target`（模型约定值，计算用）/
`total_current` / `source`（1放/0充/-1待机）。
查无数据 → 同形状字典（p_target=0, source=-1, stale=True），**两条路径键名
一致**，调用方无需判空形状。

DCDC1/2/3 充/放电功率 6 列原设计为备源，实测**全库恒 0（死列）**已注释停用；
PCS 状态位同为死列，无法用状态位判向 → 功率符号+电流方向+SOC_HMI 轨迹三重
验证是唯一可行判定链。

#### ③ get_SoC(as_of, max_stale_sec=40, mismatch_pct=5.0) → (soc|None, info)

- 同行取 `SOC_CPU Calculation` + `SOC_HMI Calculation`（%）
- 归一化 ÷100 → 0~1（模型口径，换算只在此处发生一次）
- 校验：范围 [0,100]；CPU vs HMI 偏差 > 5% 打 mismatch 标志
- **无效时返回 None**（而非数值兜底）→ 调用方保持模型积分 SoC（优雅降级；
  SoC 捏造数值比缺数值危险）
- ⚠ **当前仍读 SOC_CPU 为主值——该列全库恒 0（死列），SOC_HMI 才是活值**
  （实测 0→65.6% 正常变化）。现状导致每轮锚定 soc=0.0 → 放电状态被
  `SoC>SoC_min` 检查吞掉。修复方案（CPU>0 优先、否则 HMI）见第十六章 P0-2。

#### ④ Weather_data（energy 库）

`get_weather_data(start,end)` / `save_data(end_date=...)`：Open-Meteo 历史
天气落地到 `weather_data` 表（on_conflict='update'），live_runner 每轮调用
save_data 补齐至当日。

### 10.4 BMS 表字段活性普查结果（2026-09-09 实测）

**活字段（可用）**：

| 字段 | 单位 | 实测范围 | 用途 |
|---|---|---|---|
| PCS Real-Time Power | kW | [−19.7, +31.2] | ⭐ 功率主源（正=充/负=放） |
| Total Voltage | V | [26, 129] | 电堆电压（交叉核验 V×I≈P） |
| Total Current | A | [−250, +171] | 电堆电流（负=充/正=放，判向） |
| PCS Battery Voltage / Current | V/A | ~721V / [−1,43.7]A | PCS 高压直流侧（721×43≈31kW 自洽） |
| PCS BUSVoltage | V | ~721 | 直流母线 |
| PCS Grid-Connected Active Power | kW | [−31.9,+0.05] | 电网侧（⚠符号与 PCS RTP 相反） |
| SOC_HMI Calculation | % | [0, 65.6] | ⭐ 唯一活的 SoC |
| 9 路电解液温度 | ℃ | 29.9~35.1 | ⭐ T_ely 锚点 |
| Circulation System Oper Status_Standby | 0/1 | 翻转 | 语义存疑（充电时=1），暂不用 |
| Electric_Meter_PCS 分表三相功率/电流 | kW/A | ~11kW×3 | PCS 交流侧旁证 |
| Electric_Meter_Main_Meter 分表 | kW/A | ~80kW×3/800A | 全站级（非电池专属） |

**死字段（全库恒 0，接入前必须确认修复）**：SOC_CPU Calculation、
DCDC1/2/3 充/放电功率 ×6、DCDC1/2/3 Operating Status/Charge-Discharge Type/
Operating Mode/Step Type、PCS Operating Status、PCS Battery Operating Status、
PCS Battery Charging Status、PCS Standby、PCS Charging/Discharging Energy Total、
PCS BUSCurrent、Chiller Operating Status、Energy Storage Module 1 Operating Status。

---

## 十一、实时运行层（live_runner.py）★

### 11.1 主循环时序（每轮 = 1 小时）

```
now      = start_dt + h小时          # 目标整点（决策生效时刻）
ready_at = now − lead_minutes(5)     # 计算锚点
realtime 且 h>0 → _sleep_until(ready_at)          # 真·实时: 睡到整点前5分钟
realtime 且已越过整点 → 打印 slot_late 时效告警
as_of = ready_at                                    # ★ 唯一时间锚点
  ├ current_T = get_current_max_temp(as_of)         # 温度锚（实测9测点max）
  ├ Weather_data().save_data(end_date=as_of日)      # 天气历史落地
  ├ p_info    = get_real_power(as_of)               # 功率锚（含符号翻译）
  └ soc_meas  = get_SoC(as_of)                      # SoC锚（None→退回模型积分值）
      soc_meas 有效 → soc=soc_meas; soc≥0.9 时同步 charge_latch
P_window = get_dispatch_target(now.hour, p_meas=p_info['p_target'])
res      = run_once(now, meas, ctx, P_target=P_window)   # 七步滚动
action   = res['first_action']                          # 承诺首步(0/1)
_settle_hour(...)   # 模型结算本小时成本、推进SoC/锁存(下一轮被实测重新锚定)
打印决策链 + _print_hour 摘要
```

**同步设计核心**：温度/功率/SoC/天气全部以同一个 `as_of=ready_at` 截止取数，
三路实测是同一时刻快照（功率/SoC 同一行，温度同一窗口）；优化计算耗时不影响
同步性（输入都盖了 ready_at 时间戳）。回测时 ready_at=仿真时钟 → 全链路
跟随指定日期；实时时 ready_at=真实 now−5min → 保持"提前5分钟备好指令"语义。

### 11.2 SoC 双轨制

- **锚**：每轮实测 SoC（数据库）覆盖 → 积分误差每小时清零
- **帆**：轮内 24h 推演仍靠模型法拉第积分（DP 需要完整轨迹）
- 实测无效（None）→ 整轮退回纯模型积分值（优雅降级）

### 11.3 _settle_hour 结算

用物理模型对承诺动作结算本小时：冷却电费、SoC 推进（`P_target=P_window[0]`
实测值，部分负载口径）、锁存更新。⚠ 与 DP 内部（额定功率口径）不一致 →
第十六章 P1-2。

### 11.4 输出产物

- `output/live_rolling_{YYYYMMDD}.csv`：逐时 11 列（时刻/动作/温度/SoC/电价/
  成本/覆写标志等）
- `output/live_{日期}_{1..5}_*.png`：温度、计划甘特、电价、电费、SoC 五张独立图
  （配色统一：谷绿/峰橙/尖峰红；关键点数值标注；面向运维/汇报可独立截取）

---

## 十二、命令行入口（main_rolling.py）

运行目录必须在 `脚本\`（父目录），用 `-m` 模块方式（项目为相对导入）：

```bash
cd C:\Users\user\Desktop\work_documents\脚本

# ① 实时模式（生产形态）：每整点前5分钟计算，整点前备好 0/1 指令
python -m cooling_scheduler_copy_copy_new_date_new.main_rolling --live --realtime --lead-minutes 5

# ② 快速演播：从当前整点起 24 轮连续跑完（不等时钟，开发验证用）
python -m cooling_scheduler_copy_copy_new_date_new.main_rolling --live

# ③ 指定日期回测：全链路使用指定日期的数据库实测数据（自动 realtime=False, lead=0）
python -m cooling_scheduler_copy_copy_new_date_new.main_rolling --live --once "2026-09-05 00:00"

# 可选初始状态（SoC 实测无效时的兜底）
... --soc 0.5 --prev-action 0 --latch 0
```

- `--once "YYYY-MM-DD HH:MM"` 与 `--live` 联用 = 离线回测（天气自动切 archive
  历史 API，电价按当日查库，实测三锚点按仿真时钟取数）
- 单步模式（`--once` 不带 `--live`）当前已整段注释停用
- **电价采集子进程**：main() 启动时自动调用 `start_power_price_collector()`，
  以 `subprocess.Popen([sys.executable, "-X","utf8","-u", ...])` 非阻塞拉起
  `../data_collection/power_price.py`（Playwright+Edge 爬广东电力交易中心
  地图电价 → energy.electricity_price_v3）。日志 `output/power_price.log`；
  stdin=DEVNULL、CREATE_NO_WINDOW；采集失败不影响调度（电价有三级回退）。
  `-X utf8` 必须保留：子进程 stdout 重定向后默认 GBK，脚本内 emoji 会触发
  UnicodeEncodeError 崩溃（已实测踩坑）。

---

## 十三、可视化（live_visualizer.py）

5 张独立图而非复合大图：面向非技术使用者（运维/汇报）可单独截取某一维度。
图 5（SoC）带充放电时段底色。复合总览图由 tools 脚本另行生成（旧项目）。

---

## 十四、全局配置参数（config.py）

`SystemConfig` 按功能分组（_build_xxx 各自独立）：insulation / solar / phys /
stack / pump / hex / cool / heat / agitator / price / sim / optimizer / rolling。

### 14.1 关键参数摘录（当前值）

| 参数 | 值 | 说明 |
|---|---|---|
| stack.P_nom | **42,000 W** | 额定功率（旧项目 45kW，本项目已改 42kW） |
| stack.Cap_nom | 270,000 Wh | 标称容量（⚠ 现场反推 ~190-210kWh，见十五章） |
| stack.SoC_min / max | 0.10 / 0.90 | SoC 窗口 |
| stack.start_charge_h / start_disch_h | 10.0 / 18.0 | 兜底时间窗（P_target=None 时生效；充电窗止 17:00、放电窗止 6:00 硬编码于 thermal_model） |
| stack.R_ref / Ea_R | 0.015Ω / 2500 | Arrhenius 内阻（⚠ 未对本机辨识） |
| stack.U_OCV_ref | 55 V | 基准开路电压（⚠ 现场实测堆压 ~120V） |
| stack.dS_coeff | 0.05 | 可逆熵热系数 |
| stack.P_shunt_W | 450 W | 100%SoC 旁路损耗 |
| pump.ratio_rated / ratio_idle | 0.049 / 0.007 | 泵功率 = P_nom × 比例 |
| hex.plate_UA / plate_COP_ref | 544 W/K / 2.9 | 板式换热器 |
| phys.C_th | 60.48 MJ/K | 电解液热容（14000L×1.35×3200） |
| optimizer.T 网格 | 20~40℃ / 0.25 | 81 格 |
| optimizer.SoC 网格 | 0~1 / 0.05 | 21 格（优化器内硬编码） |
| optimizer.T_target_low/high | 25 / 35 ℃ | 舒适区（软约束） |
| optimizer.T_dyn_base / slope | 38 / 4 | 动态上限 = 38−4×SoC |
| optimizer.startup_penalty | 3 元/次 | 启停惩罚 |
| optimizer.horizon_h | 24 | 优化时域 |
| rolling.commit_horizon_h | 1 | 冻结承诺时域 |
| rolling.T_emergency_high / T_safe_low | 38 / 18 ℃ | 安全覆写阈值 |
| price 时段表 | 谷00-08 / 平08-10 / 峰10-12 / 平12-14 / 峰14-19 / 平19-24 | 平=0.47×峰+0.53×谷；尖峰=峰×1.25（定义未触发） |
| DB_CONFIG | energy 库 + mqtt 库连接 | ⚠ 密码明文（第十六章 P2-8） |

死参数（收了没用/历史遗留）：sim.days（thermal_model 固定 24h）、
stack.active、hex.active、optimizer.run_penalty（已注释）、
peak_price_threshold、main_rolling 的 --days/--seeds。

---

## 十五、现场实测数据与模型参数对照 ★

2026-09-08/09 现场数据接入后获得的实测对照（辨识工作的输入）：

| 量 | 模型值 | 现场实测 | 偏差与影响 |
|---|---|---|---|
| 电堆电压 | U_OCV_ref=55V | **119~129V**（充电 125V/放电 119V） | ~2.2倍；影响电流反解、I²R 产热、SoC 积分计量 |
| 电池容量 | Cap_nom=270kWh | 反推 **~190-210kWh**（9-9 充电 3.75h×29kW≈109kWh交流 → SOC 0→53.5%） | ~-25%；影响 SoC 轨迹斜率 |
| 采样间隔 | 假设 5s | **6~8s/条**（正常时段 ~540条/小时） | 影响健康度判断基准（近10分钟 ≈90-120 条为正常） |
| PCS 功率待机噪声 | — | **达 −0.64kW**（P1=−0.50kW，54,524 待机样本） | EPS_W 定为 1000W 的依据 |
| 总电流待机噪声 | — | −0.36~−0.69A | EPS_A=2A 的依据（>噪声，<最小工况 146A） |
| 充电功率 | 计划表 −25kW | 实际 +24~31kW（PCS 符号） | 幅值接近，符号需翻译 |
| 现场调度时段 | 计划表：0-7充/10-11,14-18放 | 实际 9-9：**11:00-15:30+ 充电** | 计划表与现场不符（P0-3） |
| 数据流稳定性 | 假设连续 | **48h 内断流 4 次**（最长 18.7h；恢复点多在 12:26 前后） | fallback/stale 机制常触发；根因待查（入库服务/broker/网关） |

---

## 十六、已知问题与改进路线图

### P0 — 不解决不能投运

| # | 问题 | 位置 | 说明 |
|---|---|---|---|
| P0-1 | **数据流反复断流** | 现场链路 | 48h 断 4 次；断流时温度 fallback 28℃（与真实差 5-7℃，足以翻转开关决策）。排查顺序：入库服务日志 → MQTT broker（mosquitto_sub 订阅验证）→ 网关。DB 连接层已有 _fetchone 重连兜底 |
| P0-2 | **get_SoC 仍读死列 SOC_CPU** | save_database.get_SoC | SOC_CPU 全库恒 0 → 每轮锚定 soc=0.0 → 放电状态被 SoC>SoC_min 吞掉、dynamic_T_max 恒最宽松。改为 CPU>0 优先、否则 SOC_HMI（活值，实测 0~65.6%） |
| P0-3 | **硬编码计划表与现场调度不符** | live_runner.get_dispatch_target | 表：14-18 点放电；实际 11-15 点充电。尾部 23 格推演基础错误，预冷套利失效。短期按运维实际时段改数组；中期用 DB 历史统计"典型日功率曲线"自动刷新；长期接 EMS 计划 |
| P0-4 | **控制指令无出口** | — | first_action 只打印/写CSV，未接 PLC（Modbus/MQTT publish/OPC-UA）。lead_minutes=5 的"提前备指令"设计等待出口落地 |

### P1 — 投运后第一个月

| # | 问题 | 说明 |
|---|---|---|
| P1-1 | EPS 现场标定 | EPS_W=1000/EPS_A=2 为数据驱动初值；投运后用待机噪声 P99×2 与最小工况功率之间复核 |
| P1-2 | **optimizer 两处 step_hourly 未传 P_target（L283/L409 被注释）** | 状态判断用实测功率、产热/SoC 积分按额定 42kW → 部分负载时产热高估~2倍；且与 _settle_hour（传了）口径分裂。真实数据已接入，应打开注释 |
| P1-3 | 模型参数辨识 | 用 DB 实测做 C_th/UA/R_ref/U_OCV_ref/Cap_nom 离线辨识（十五章偏差表为起点）；55V→120V 与 270→200kWh 是当前最大系统性误差源 |
| P1-4 | live 模式结果入库 | run_live 未调 DB_RollingResult，事后无法 SQL 复盘每轮决策 |
| P1-5 | 进程守护与状态恢复 | 崩溃/断电后 prev_action/latch/承诺表丢失；建议每轮状态写库 + 启动时恢复，或任务计划守护 |
| P1-6 | 新增取数函数零测试 | get_real_power/get_SoC/_state_from_power 需补单测（正常/断流/坏值/符号四场景） |

### P2 — 工程质量

| # | 问题 |
|---|---|
| P2-1 | DB 密码明文（config.py、power_price.py、工具脚本多处）→ 环境变量/密钥管理 |
| P2-2 | 死代码与半停用状态：单步模式整段注释、--t-current 注释、DCDC 备源注释、get_real_power 内调试 print("从数据库中拿到了数据…") 每轮刷屏 |
| P2-3 | print 无 logging：实时长跑无落盘日志、无级别控制 |
| P2-4 | 充/放电窗止点（17:00/6:00）硬编码 thermal_model 两处；SoC 网格 21 格硬编码 optimizer |
| P2-5 | DP 温度网格 20~40℃ 越界静默钳位，无断言告警 |
| P2-6 | 承诺冻结/安全覆写为事后修正而非 DP 约束（value function 与真实可行集不一致） |
| P2-7 | 纯确定性优化：无天气场景/鲁棒处理；惩罚权重（10/5/20/3元）未标定无敏感性分析 |
| P2-8 | 目标函数缺电池退化成本（高温运行寿命代价）；fan_state/加热滞环未入 DP 状态 |
| P2-9 | 尖峰电价档定义未触发；EKES COP 不随 scale 缩放（沿自旧项目） |
| P2-10 | query_power_range.py 的 `END=datetime.now()` 为 naive 时间，跨时区服务器可能偏 8h（建议带 CN_TZ） |

---

## 十七、验证记录

| 日期 | 验证项 | 结果 |
|---|---|---|
| 2026-09-07 | get_real_power 7 列取数 + Excel 导出（test_real_power.py） | ✅ 链路通；当时功率全 0（系统停机） |
| 2026-09-08 | as_of=ready_at 日期同步改造 | ✅ 指定 9-05 回测 24 轮，数据时刻逐轮走在 9-05，温度 31.1~31.8℃ 实测锚定，累计电费 23.59 元，commit_hold 覆写正常触发 |
| 2026-09-08 | 首次实测到非零功率 PCS=−19.65kW | 触发主源从 DCDC(死列)切换为 PCS 的设计决策 |
| 2026-09-09 | **符号约定验证**：SOC_HMI 上升段 vs PCS 功率符号交叉表 | ✅ 224 样本 100% 对应（SOC升↔PCS正），0 反例 → 数据库"正=充电"实锤，get_real_power 加 p_target=-pcs_w 翻译 |
| 2026-09-09 | 充电会话 11:22-15:30+，24~31kW，SOC_HMI 0→55% | 实测容量反推 ~190-210kWh；堆压 ~125V |
| 2026-09-09 | 子进程 GBK/emoji 崩溃 | ✅ -X utf8 + reconfigure 双保险修复，复现场景验证通过 |
| 待做 | 放电符号验证（放电时 PCS 应为负）| 等现场下次放电会话确认 |
| 待做 | 实时模式全链路（数据流稳定 + P0-2 修复后） | — |

---

## 十八、附录：常用 SQL 与工具脚本

### 18.1 数据流健康检查

```sql
SELECT max(timestamp) AS 最新数据,
       count(*) FILTER (WHERE timestamp >= now() - interval '10 minutes') AS 近10分钟条数
FROM "BMS";     -- 近10分钟 ≈90-120 条正常；<10 条 = 断流
```

### 18.2 断流缺口定位（最近48h）

```sql
WITH t AS (SELECT timestamp, lag(timestamp) OVER (ORDER BY timestamp) AS prev
           FROM "BMS" WHERE timestamp >= now() - interval '48 hours')
SELECT prev AS 断流开始, timestamp AS 恢复, timestamp - prev AS 时长
FROM t WHERE timestamp - prev > interval '60 seconds' ORDER BY prev;
```

### 18.3 指定时间范围原始数据

```sql
SELECT timestamp, "PCS Real-Time Power", "Total Current", "SOC_HMI Calculation"
FROM "BMS"
WHERE timestamp >= '2026-09-09 11:00:00+08' AND timestamp <= '2026-09-09 15:30:00+08'
ORDER BY timestamp;
-- 注意: 表名/含空格列名必须双引号; 时间字面量必须带 +08 时区后缀
```

### 18.4 工具脚本

- `test_real_power.py`（本项目内）：7 功率列取数测试 + Excel 四 sheet 导出
- `../query_power_range.py`：改 START/END 两行 → 任意列组合原始数据导出 CSV
- `python -m cooling_scheduler_copy_copy_new_date_new.save_database`：
  Real_time_db 三函数冒烟测试（__main__ 块）

---

*本文档基于 2026-09-09 代码现状与现场数据库实测整理；符号约定、字段活性、
参数偏差均有 SQL 验证记录支撑（第十七章）。现场设备与数据流仍在联调阶段，
第十六章 P0 项完成前，实时模式输出仅有链路验证价值，不具备调度参考价值。*
