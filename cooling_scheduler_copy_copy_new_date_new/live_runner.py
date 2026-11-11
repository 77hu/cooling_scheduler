# ============================================================================
# cooling_scheduler_copy/live_runner.py
# 模块 14：真实数据·持续滚动运行器（--live）
#
# 目标：真数据 + 一直挂着跑 + 每小时输出开/关与成本。
#
# 三类数据来源：
#   · 电解液温度（每轮锚点）  → 用户提供的 24h 实测数据（--temp-file）
#   · 环境天气（换热用）      → Open-Meteo 真实预报（fetch_weather_window）
#   · 电价                    → 数据库真实价（resolve_price_pairs，失败回退默认）
#
# 每轮逻辑：
#   第 h 轮：当前温度=温度数据[h] → 拉真天气+真电价 → 滚动优化 →
#           定第 h 小时开/关 → 结算本小时成本+累计 → 推进 SoC/锁存 → 下一轮
#

# 依赖：模块 1/2/3/5/11
# ============================================================================

import os
import csv
import time
import datetime

import numpy as np

from .config import SystemConfig,CN_TZ
from .thermal_model import DetailedThermalModel
from .optimizer import CoolingOptimizer
from .rolling_runner import MeasuredState, RunContext, run_once
from .save_database import Real_time_db,Weather_data,DB_RollingResult


# ============================================================================
# 温度数据读取
# ============================================================================

def load_temperature_data(as_of=None, db=None):
    if db is None:
          db = Real_time_db()        
    return db.get_current_max_temp(as_of=as_of)


# ============================================================================
# 主流程
# ============================================================================

def run_live(start_dt=None, realtime=False,
             initial_soc=0.5,
             prev_action=0, latch=0, out_dir="output", save_csv=True,
             lead_minutes=5,P_target=None,save_db=True):
    """真实数据·持续滚动运行。

    参数:
        start_dt:    起始时刻（默认当前整点）
        realtime:    True=真·实时（每整点跑一轮）；False=快速演播
        initial_soc / prev_action / latch: 第 0 轮初始状态
        out_dir:     CSV 输出目录
        save_csv:    是否写 CSV
    """
    # temp_data = load_temperature_data()
    # n_hours = len(temp_data)
    rt_db = Real_time_db() 
    n_hours = 24 

    print("\n" + "=" * 65)
    print("  冷却滚动 · 真实数据持续运行 (--live)")
    # print(f"  温度数据: 内置占位曲线（{n_hours} 小时）")
    print(f"  运行模式: {'真·实时（每整点一轮）' if realtime else '快速演播（连续跑完）'}")
    print("=" * 65)

    config = SystemConfig()
    model = DetailedThermalModel(config)
    optimizer = CoolingOptimizer(config)
    # RunContext 默认 fetch_weather_window / resolve_price_pairs = 真实数据源
    ctx = RunContext(config=config, model=model, optimizer=optimizer)

    # ---- 结果入库（fail-operational: 连不上只写CSV,不阻塞控制）----
    db_res = None
    if save_db:
        try:
            db_res = DB_RollingResult(config)
            db_res.create_rolling_tables()
        except Exception as e:
            print(f"  ⚠️ 结果入库连接失败（本次只写CSV）: {e}")
            db_res = None
    prev_applied_rec = None      # 上一小时的applied记录,待本轮回填真值
    _nn = lambda v: (round(float(v), 2) if v is not None and v == v else None)   # NaN→None, 统一保留2位小数
    # ---- 天气历史落地连接: 全程复用一个实例(此前每轮new一个且从不关闭,泄漏连接打爆DB) ----
    wd = None
    try:
        wd = Weather_data()
    except Exception as e:
        print(f"  ⚠️ Weather_data 连接失败（跳过天气落地,不阻塞控制）: {e}")

    lead = max(0, min(59, int(lead_minutes)))

    if start_dt is None:
        now_wall = datetime.datetime.now(CN_TZ).replace(tzinfo=None)
        start_dt = now_wall.replace(minute=0, second=0, microsecond=0)
        # ---- 启动对齐 ----
        # 若当前时刻距整点已不足 lead 分钟，来不及按设计提前备好本整点指令，
        # 该时段实际已过期/即将过期，跳过它，从下一个整点开始，
        # 避免下发"到期即作废"的冷却指令（如 16:59 启动却先出 16 点计划）。
        if now_wall >= start_dt + datetime.timedelta(minutes=60 - lead):
            print(f"  [启动对齐] 当前 {now_wall:%H:%M}，距下一整点不足 {lead} 分钟，"
                  f"为 {start_dt:%H:%M} 时段备指令已来不及 → 跳过该时段，"
                  f"从 {start_dt + datetime.timedelta(hours=1):%H:%M} 开始")
            start_dt += datetime.timedelta(hours=1)

    soc = float(initial_soc)
    prev_a = int(prev_action)
    latch_state = int(latch)
    cum_cost = 0.0
    logs = []
    last_wd_date = None          # 天气落地上次执行的日期（每天只做一次,不必每轮全量刷）
    for h in range(n_hours):
        now = start_dt + datetime.timedelta(hours=h)
        ready_at = now - datetime.timedelta(minutes=lead)
        if realtime and h > 0:
            _sleep_until(ready_at)   # 提前 lead 分钟醒来计算、整点前备好指令
        if realtime:
            lag_min = (datetime.datetime.now(CN_TZ).replace(tzinfo=None) - now).total_seconds() / 60.0
            if lag_min > 0:
                print(f"  ⚠ 时效告警[slot_late]: 当前已越过目标整点 {now:%H:%M} "
                    f"{lag_min:.0f} 分钟，本时段指令迟发（数据求解/进程阻塞所致？）")
        # current_T = float(temp_data[h])
        # current_T = load_temperature_data(as_of=ready_at, db=rt_db)
        # as_of = datetime.datetime.now()                              # 真实计算时刻
        as_of=ready_at
        current_T = load_temperature_data(as_of=as_of, db=rt_db)
        if wd is not None and as_of.strftime("%Y-%m-%d") != last_wd_date:
            # 每天落地一次,只拉近3天滚动窗口(archive对最近1-2天有延迟,3天窗口保证补齐)
            try:
                wd.save_data(start_date=(as_of - datetime.timedelta(days=3)).strftime("%Y-%m-%d"),
                             end_date=as_of.strftime("%Y-%m-%d"))
                last_wd_date = as_of.strftime("%Y-%m-%d")
            except Exception as e:
                print(f"  ⚠️ 天气落地失败(跳过): {e}")

        #拿到真实功率和SoC数据
        p_info=rt_db.get_real_power(as_of=as_of)
        soc_meas,s_info=rt_db.get_SoC(as_of=as_of)
        if soc_meas is not None:
            soc=soc_meas
            latch_state=model.initial_charge_latch(soc) if soc>=0.9 else latch_state  
            print(f"使用数据库的SoC：{soc}")
        else:
            print(f"使用默认初始化的SoC:{soc}")

        print(f"  [第{h}轮] 整点={now:%H:%M} ready_at={ready_at:%H:%M} "
        f"计算时刻={as_of:%H:%M:%S} → 锚点温度 T_ely={current_T:.2f}℃")
        ts_str = (p_info['data_ts'].strftime('%m-%d %H:%M:%S')
                    if p_info['data_ts'] else '无数据')
        print(f"          实测功率 P_target={p_info['p_target']:+.0f}W"
                f" [{'放电中' if p_info['source']==1 else ('充电中' if p_info['source']==0 else '待机')}]"
                f" (PCS={p_info['pcs_kw']:+.2f}kW 电流={p_info['total_current']:+.2f}A)"
                f" 数据时刻={ts_str}"
                f"{'  ⚠数据过期' if p_info['stale'] else ''})")

        # ---- 回填上一执行小时的真值（本轮锚点温度 = 上小时结束温度）----
        if db_res is not None and prev_applied_rec is not None:
            try:
                prev_applied_rec["meas_T_after"] = round(float(current_T), 2)
                db_res.save_applied_hour(prev_applied_rec)   # DO UPDATE只更新真值列
            except Exception as e:
                print(f"  ⚠️ 回填上小时真值失败: {e}")
        
        meas = MeasuredState(T_ely=current_T, 
                             SoC=soc,
                             prev_cool_action=prev_a,
                             charge_latch=latch_state)

        # ---- 外部调度功率目标：未显式传入时用接口占位 ----
        #获得任意时间的功率序列
        if P_target is None:
             P_window = get_dispatch_target(now.hour, p_meas=p_info['p_target'])
        else:
            P_window = [P_target[(h + k) % len(P_target)] for k in range(24)]

        # ---- 滚动优化（真天气 + 真电价 + 外部调度）----
        res = run_once(now, meas, ctx, P_target=P_window)
        # print("run_once返回的数据：",res)
        action = int(res["first_action"])
        opt = res.get("opt_result")
        # if opt is not None:
        #     print(f"  [调试] T_forecast: {opt['T_forecast']}")

        # ---- 用承诺动作结算本小时成本 & 推进 SoC/锁存 ----
        cost_h, soc_next, latch_next = _settle_hour(
            model, res, now.hour, current_T, soc, latch_state,
            meas.fan_state, action, P_target=P_window[0])

        price0 = (float(res["price_hourly"][0])
                  if res["price_hourly"] is not None else float("nan"))
        amb0 = (float(res["weather"]["temperature"][0])
                if res.get("weather") is not None else float("nan"))
        cum_cost += cost_h

        # ---- 本轮结果入库: 只存 applied(本执行小时台账), 不存 runs(24h预测快照) ----
        if db_res is not None:
            try:
                run_ts_str = now.strftime("%Y-%m-%d %H:%M:%S+08")
                tariff = config.get_tariff_label(now.hour)
                # status 标签派生逻辑与 cooling_schedule_result 的 _derive_status_labels 一致
                if action:
                    status_lbl = ("Pre-cool" if tariff == "谷" else
                                  ("On-demand" if tariff in ("峰", "尖峰") else "Running"))
                else:
                    status_lbl = "Standby"
                prev_applied_rec = {
                    "hour_start": run_ts_str, "run_ts": run_ts_str,
                    "action": action,
                    "action_source": (res["overrides"][0] if res["overrides"] else "new_plan"),
                    "meas_T_before": round(float(current_T), 2), "meas_T_after": None,  # 下轮回填
                    "ambient": _nn(amb0), "truth_ambient": None,
                    "truth_price": _nn(price0), "cost_hourly": _nn(cost_h),

                    "price": _nn(price0),
                    "tariff_label": tariff,
                    "raw_action": (int(res["raw_first_action"])
                                   if res["raw_first_action"] is not None else None),
                    "forecast_temp": (round(float(opt["T_forecast"][1]), 2) if opt is not None else None),
                    "soc_before": round(float(soc), 2), "soc_after": round(float(soc_next), 2),
                    "p_target_w": _nn(p_info.get("p_target")),
                    "power_source": {1: "discharge", 0: "charge",
                                     -1: "standby"}.get(p_info.get("source"), "fallback"),
                    "cum_cost": round(cum_cost, 2),
                    "irradiation": (round(float(res["weather"]["radiation"][0]), 2)
                                    if res.get("weather") is not None else None),
                    "status": status_lbl,
                    "power": (round(float(opt["P_cool_elec"][0]) / 1000.0, 2)
                              if opt is not None else None), 
                }
                db_res.save_applied_hour(prev_applied_rec)
            except Exception as e:
                print(f"  ⚠️ 入库失败（跳过，不阻塞控制）: {e}")

        dyn_T = model.dynamic_T_max(soc) if hasattr(model, "dynamic_T_max") else float("nan")
        _print_hour(h, now, ready_at, action, current_T, soc, amb0, dyn_T,
                    price0, cost_h, cum_cost, opt, config, res)

        logs.append({
            "idx": h, "abs_dt": now.strftime("%Y-%m-%d %H:%M"),
            "day_h": now.hour, "action": action,
            "T_ely": round(current_T, 3), "SoC": round(soc, 4),
            "price": round(price0, 4) if price0 == price0 else "",
            "ambient": round(amb0, 3) if amb0 == amb0 else "",
            "step_cost": round(cost_h, 4), "cum_cost": round(cum_cost, 4),
            "overrides": "|".join(res["overrides"]) if res["overrides"] else "",
        })

        soc = soc_next
        prev_a = action
        latch_state = latch_next

    print("\n" + "=" * 65)
    print(f"  完成 {n_hours} 轮 │ 累计冷却电费 {cum_cost:.2f} 元")
    print("=" * 65)

    if save_csv:
        _save_csv(logs, out_dir, start_dt)

    # ---- 生成 5 张独立可视化图 ----
    try:
        from .live_visualizer import plot_live_figures
        paths = plot_live_figures(logs, config=config, out_dir=out_dir)
        print("  可视化: " + ", ".join(paths.values()))
    except Exception as e:
        print(f"  可视化生成失败（不影响数据）: {e}")

    # ---- 关闭结果入库连接 ----
    if db_res is not None:
        try:
            db_res.close()
        except Exception:
            pass
    if wd is not None:
        try:
            wd.close()
        except Exception:
            pass

    return logs


# ============================================================================
# 辅助
# ============================================================================

def _settle_hour(model, res, hour, current_T, soc, latch_state,
                 fan_state, action,P_target=None):
    """用承诺动作结算本小时成本，并推进 SoC/锁存。

    温度下一轮由用户数据覆盖，这里只用物理模型推 SoC/锁存/成本。
    """
    if res["weather"] is None or res["price_hourly"] is None:
        return 0.0, soc, latch_state
    cds = model.determine_charge_discharge_state(hour, soc, latch_state,P_target=(P_target if P_target is not None else None))
    state0 = {"T_ely": current_T, "T_stack": current_T, "SoC": soc,
              "Current_State": cds, "fan_state": int(fan_state),
              "charge_latch": int(latch_state)}  
    w0 = {"temperature": float(res["weather"]["temperature"][0]),
          "radiation": float(res["weather"]["radiation"][0])}
    next_state, metrics = model.step_hourly(
        state0, hour, w0, float(res["price_hourly"][0]),
        cool_on=bool(action), heat_on=False, P_target=P_target)
    return (float(metrics.get("cost", 0.0)),
            float(next_state["SoC"]),
            int(next_state.get("charge_latch", latch_state)))


def _print_hour(h, now, ready_at, action, current_T, soc, amb0, dyn_T,
                price0, cost_h, cum_cost, opt, config, res):
    tag = config.get_tariff_label(now.hour)
    onoff = "开" if action == 1 else "关"
    ov = (" │ 覆写:" + ",".join(res["overrides"])) if res["overrides"] else ""
    amb_s = f"{amb0:.1f}°C" if amb0 == amb0 else "  -"
    dyn_s = f"{dyn_T:.1f}°C" if dyn_T == dyn_T else "  -"
    print(f"\n  当前时间为 {ready_at:%m-%d %H:%M} ---> 预测时间计划安排 {now:%m-%d %H:%M}")
    print(f"━━━ [{now:%Y-%m-%d %H:%M}] 第 {h} 轮 ━━━")
    print(f"  本小时: 冷却={onoff} │ 电解液 {current_T:.1f}°C │ 环境 {amb_s} │ "
          f"动态上限 {dyn_s} │ SoC {soc:.2f}")
    print(f"          电价 {price0:.3f}元({tag}) │ 本时电费 {cost_h:.2f}元 │ "
          f"累计 {cum_cost:.2f}元{ov}")
    if opt is not None:
        sched = opt["schedule"]
        plan_str = "".join("开" if x else "·" for x in sched)
        print(f"  未来24h计划: {plan_str}    (首刻：{now:%H}:00)")
        print(f"  窗口预计总电费: {opt['total_cost']:.2f} 元 │ "
              f"冷却 {int(np.sum(sched))} h │ "
              f"风扇电费 {opt.get('Cost_fan_CNY', 0.0):.2f}元 / "
              f"总 {opt.get('Cost_total_CNY', 0.0):.2f}元")
    else:
        print("  （无计划：数据缺失保持模式）")


def _sleep_until(target_dt):
    """真·实时：睡到目标时刻。"""
    while True:
        remain = (target_dt - datetime.datetime.now(CN_TZ).replace(tzinfo=None)).total_seconds()
        if remain <= 0:
            break
        time.sleep(min(remain, 30))


def _save_csv(logs, out_dir, start_dt):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"live_rolling_{start_dt:%Y%m%d}.csv")
    cols = ["idx", "abs_dt", "day_h", "action", "T_ely", "SoC", "price",
            "ambient", "step_cost", "cum_cost", "overrides"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in logs:
            w.writerow(row)
    print(f"\n  ✅ 逐时记录已写入 {path}")


def get_dispatch_target(start_hour,n=24,p_meas=None):
    """
    外部调度功率目标接口[w]。
    约定：>0=放电，<=0充电，=0=待机；返回长度n，索引0=窗口起始小时。
    
    """
    P=[
      -35000, -35000, -35000, -35000, -35000, -35000, -35000, -35000,  # 0-7点  谷时充电 25kW
      0,      0,                                                        # 8-9点  待机
      35000,  35000,                                                    # 10-11点 峰时放电 30kW
      0,      0,                                                        # 12-13点 待机
      35000,  35000,  35000,  35000,  35000,                            # 14-18点 峰时放电 30kW
      0,      0,      0,      0,      0,                                # 19-23点 待机
    ]
    #可获得任意时间的功率序列
    windows= [P[(start_hour + k) % 24] for k in range(n)]
    #接入真实外部调度功率计划可替代
    if p_meas is not None:
        windows[0]=float(p_meas)
    return windows