# ============================================================================
# cooling_scheduler_copy/rolling_runner.py
# 模块 11：小时级滚动时域调度核心（MPC 式）
#
# 职责：每小时一次滚动重优化——
#   1. 读取当前实测状态（T_ely / SoC / 上步冷却动作 / 充电锁存）；
#   2. 取最新滑动窗口预报（未来 24h 天气）与电价（三级回退，可跨午夜）；
#   3. 用 DP 重解完整 24h 计划；
#   4. 只执行（承诺）第一小时决策，登记后续冻结承诺；
#   5. 数据缺失时 fail-operational：沿用上一轮承诺动作。
#
# 设计要点：
#   - 天气/电价数据源可注入（RunContext），闭环仿真与单元测试用伪实现；
#   - 安全覆写（超温强开 / 低温强关）永远压过冻结承诺；
#   - charge_latch 跨运行显式传递（锁存不能靠 SoC 反推：置位后 shunt
#     会把 SoC 拖到 SoC_max 以下，出现"SoC<上限但锁仍生效"的合法态）。
#
# 依赖：模块 1（config）、模块 2（data_fetcher）、模块 3（thermal_model）、
#       模块 5（optimizer）
# 被依赖：模块 12（闭环验证平台）、main_rolling（入口）
# ============================================================================

import datetime

from .config import SystemConfig
from .thermal_model import DetailedThermalModel
from .optimizer import CoolingOptimizer
from .data_fetcher import (fetch_weather_window, resolve_price_pairs,
                           build_price_hourly)


class MeasuredState:
    """某时刻的实测状态（由 BMS/SCADA 或仿真 plant 提供）。"""

    __slots__ = ("T_ely", "T_stack", "SoC", "charge_latch",
                 "prev_cool_action", "fan_state")

    def __init__(self, T_ely, SoC, prev_cool_action=0, charge_latch=0,
                 T_stack=None, fan_state=0):
        self.T_ely = float(T_ely)
        self.T_stack = float(T_stack if T_stack is not None else T_ely)
        self.SoC = float(SoC)
        self.charge_latch = int(charge_latch)
        self.prev_cool_action = int(prev_cool_action)
        self.fan_state = int(fan_state)


class RunContext:
    """滚动运行上下文：配置/模型/优化器 + 可注入数据源 + 承诺状态。"""

    def __init__(self, config=None, model=None, optimizer=None,
                 fetch_weather=fetch_weather_window,
                 resolve_pairs=resolve_price_pairs,
                 commit_horizon_h=1,
                 T_emergency_high=38.0, T_safe_low=18.0,
                 horizon_h=24):
        """
        参数:
            config/model/optimizer: 可注入复用；缺省自动构建
            fetch_weather:  (start_dt, horizon_h) -> weather dict | None
            resolve_pairs:  (dates, today) -> (pairs, tiers)
            commit_horizon_h: 冻结时长 [h]（0~2）。每轮把新计划的前
                            commit_horizon_h 小时登记为承诺；下一轮
                            若首步决策与承诺冲突则维持承诺。
            T_emergency_high: 超温强开阈值 [°C]（安全覆写）
            T_safe_low:       低温强关阈值 [°C]（安全覆写）
            horizon_h:        前瞻窗口长度 [h]
        """
        self.config = config if config is not None else SystemConfig()
        self.model = (model if model is not None
                      else DetailedThermalModel(self.config))
        self.optimizer = (optimizer if optimizer is not None
                          else CoolingOptimizer(self.config))
        self.fetch_weather = fetch_weather
        self.resolve_pairs = resolve_pairs
        self.commit_horizon_h = int(commit_horizon_h)
        self.T_emergency_high = float(T_emergency_high)
        self.T_safe_low = float(T_safe_low)
        self.horizon_h = int(horizon_h)
        # 承诺表：{绝对小时 datetime: action}，由每轮 run_once 登记/清除
        self.committed = {}


def _floor_to_hour(dt):
    """截断到整点（非整点运行按上一个整点处理）。"""
    return dt.replace(minute=0, second=0, microsecond=0)


def _safety_override(action, T_ely, ctx,SoC=None):
    """安全覆写：超温强开、低温强关。返回 (action, override 标签|None)。"""
    lim=ctx.model.dynamic_T_max(SoC) if SoC is not None else ctx.T_emergency_high 
    if T_ely>=lim and action !=1:
        return 1,"safety_on"
    if T_ely<=ctx.T_safe_low and action !=0:
        return 0,"safety_off"
    return action,None

    # if T_ely >= ctx.T_emergency_high and action != 1:
    #     return 1, "safety_on"
    # if T_ely <= ctx.T_safe_low and action != 0:
    #     return 0, "safety_off"
    # return action, None


def run_once(run_dt, meas, ctx,P_target=None):
    """
    一次滚动优化运行：重解未来 horizon_h 小时完整计划，返回承诺首步。

    参数:
        run_dt: 运行时刻（datetime，本地时间；非整点自动 floor）
        meas:   MeasuredState 实测状态
        ctx:    RunContext 运行上下文

    返回:
        dict:
          status:            "ok" | "no_data_hold"
          window_start:      窗口起始 datetime（整点）
          start_hour:        窗口起始钟表时刻 (0-23)
          weather:           窗口天气 dict | None
          price_hourly:      窗口逐时电价 ndarray | None
          price_tiers:       {date: "db"|"today_fallback"|"default"} | None
          opt_result:        optimize() 完整返回 | None
          raw_first_action:  DP 原始首步决策 | None
          first_action:      实际承诺动作（经冻结 + 安全覆写）
          overrides:         触发的覆写标签列表
                             ("no_data_hold"/"commit_hold"/"safety_on"/"safety_off")
          latch_after:       执行首步后的充电锁存状态（下一运行传回）
          solve_time_s:      求解耗时 | None
    """
    start_dt = _floor_to_hour(run_dt)
    start_hour = start_dt.hour
    H = ctx.horizon_h
    overrides = []

    # ------------------------------------------------------------------
    # 1. 天气窗口（fail-operational：失败 → 缓存 → 保持上轮承诺）
    # ------------------------------------------------------------------
    weather = ctx.fetch_weather(start_dt, H)
    if weather is None:
        hold = ctx.committed.get(start_dt)
        action = int(hold) if hold is not None else meas.prev_cool_action
        overrides.append("no_data_hold")
        action, so = _safety_override(action, meas.T_ely, ctx, SoC=meas.SoC)
        if so:
            overrides.append(so)
        return {"status": "no_data_hold", "window_start": start_dt,
                "start_hour": start_hour, "weather": None,
                "price_hourly": None, "price_tiers": None,
                "opt_result": None, "raw_first_action": None,
                "first_action": int(action), "overrides": overrides,
                "latch_after": meas.charge_latch, "solve_time_s": None}

    # ------------------------------------------------------------------
    # 2. 电价：窗口覆盖的日历日做三级回退，逐时拼接（可跨午夜）
    # ------------------------------------------------------------------
    dates = sorted({(start_dt + datetime.timedelta(hours=k)).strftime("%Y-%m-%d")
                    for k in range(H)})
    today_str = start_dt.strftime("%Y-%m-%d")
    pair_by_date, tiers = ctx.resolve_pairs(dates, today_str)
    price_hourly = build_price_hourly(start_dt, H, pair_by_date, ctx.config)
    # print("小时价格情况：",price_hourly)
    # ------------------------------------------------------------------
    # 3. DP 重解（实测状态 + 锁存跨运行接续 + 上步动作）
    # ------------------------------------------------------------------
    opt = ctx.optimizer.optimize(
        {"temperature": weather["temperature"],
         "radiation": weather["radiation"]},
        price_hourly,
        T_initial=meas.T_ely, model=ctx.model, SoC_initial=meas.SoC,
        prev_a_initial=meas.prev_cool_action,
        start_hour=start_hour, charge_latch_initial=meas.charge_latch,P_target=P_target)

    schedule = opt["schedule"]
    raw_first = int(schedule[0])
    action = raw_first

    # ------------------------------------------------------------------
    # 4. 冻结：当前小时若已有上轮承诺且与本轮冲突 → 维持承诺
    # ------------------------------------------------------------------
    hold = ctx.committed.get(start_dt)
    if hold is not None and int(hold) != action:
        action = int(hold)
        overrides.append("commit_hold")

    # ------------------------------------------------------------------
    # 5. 安全覆写（永远压过冻结；传 SoC → 用动态温度上限 38-4×SoC）
    # ------------------------------------------------------------------
    action, so = _safety_override(action, meas.T_ely, ctx, SoC=meas.SoC)
    if so:
        overrides.append(so)

    # ------------------------------------------------------------------
    # 打印本轮决策链（可审计）：上轮承诺 vs DP 原始首步 vs 最终动作
    #   hold       = 上一轮对当前小时登记的承诺（可能为 None）
    #   raw_first  = 本轮 DP 重新解出的原始首步
    #   action     = 经过【冻结承诺】和【安全覆写】之后的最终动作
    # ------------------------------------------------------------------
    hold_txt = ("开" if hold == 1 else "关") if hold is not None else "无"
    same_txt = "是" if (hold is None or int(hold) == raw_first) else "否"
    print(f"\n  [决策链] {start_dt:%Y-%m-%d %H:%M} | "
          f"上轮承诺={hold_txt} | DP本轮首步={'开' if raw_first else '关'} | "
          f"两者一致={same_txt} | 本轮最终动作={'开' if action else '关'} | "
          f"覆写={overrides if overrides else '无'}")

    # ------------------------------------------------------------------
    # 6. 登记未来 commit_horizon_h 小时的承诺，并清除过期项
    # ------------------------------------------------------------------
    if ctx.commit_horizon_h > 0:
        for j in range(1, 1 + ctx.commit_horizon_h):
            if j < H:
                ctx.committed[start_dt + datetime.timedelta(hours=j)] = \
                    int(schedule[j])
    ctx.committed = {t: a for t, a in ctx.committed.items() if t > start_dt}

    # ------------------------------------------------------------------
    # 7. 执行首步后的锁存状态（用模型重推一步，严格于任何近似）
    # ------------------------------------------------------------------
    cs0 = ctx.model.determine_charge_discharge_state(
        start_hour, meas.SoC, meas.charge_latch,P_target=(P_target[0] if P_target is not None else None))
    
    state0 = {"T_ely": meas.T_ely, "T_stack": meas.T_stack,
              "SoC": meas.SoC, "Current_State": cs0,
              "fan_state": meas.fan_state,
              "charge_latch": meas.charge_latch}
    next0, _ = ctx.model.step_hourly(
        state0, start_hour,
        {"temperature": float(weather["temperature"][0]),
         "radiation": float(weather["radiation"][0])},
        float(price_hourly[0]), cool_on=bool(action), heat_on=False,
        P_target=(P_target[0] if P_target is not None else None))
    latch_after = int(next0["charge_latch"])

    return {"status": "ok", "window_start": start_dt,
            "start_hour": start_hour, "weather": weather,
            "price_hourly": price_hourly, "price_tiers": tiers,
            "opt_result": opt, "raw_first_action": raw_first,
            "first_action": int(action), "overrides": overrides,
            "latch_after": latch_after,
            "solve_time_s": opt.get("solve_time_s")}


def make_default_context(**kwargs):
    """构建默认 RunContext（真实数据源；生产/单步模式用）。"""
    return RunContext(**kwargs)
