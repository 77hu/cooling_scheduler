# ============================================================================
# cooling_scheduler_copy/main_rolling.py
# 模块 13：滚动时域调度入口
#
# 依赖：模块 1/2/11/12、save_database（可选）
# ============================================================================

import sys
import os
import argparse
import datetime
import subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import threading
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from .config import SystemConfig, LOCATION_CONFIG
import time
from .rolling_runner import (MeasuredState, run_once,
                                                   make_default_context)


def print_plan_table(res, config):
    """打印本轮逐时计划表（滚动版）。"""
    opt = res.get("opt_result")
    if opt is None:
        print("\n  无计划生成（数据缺失保持模式，沿用上轮承诺动作）")
        return
    schedule = opt["schedule"]
    T_fc = opt["T_forecast"]
    P_cool = opt["P_cool_elec"]
    cost_h = opt["cost_hourly"]
    ph = res["price_hourly"]
    ws = res["window_start"]

    print("\n" + "=" * 88)
    print(f" 滚动优化计划  窗口起点 {ws:%Y-%m-%d %H:%M}"
          f"（只承诺首步：{'开' if res['first_action'] else '关'}）")
    print("=" * 88)
    print(f"  {'相对h':>5} {'时刻':>11} │ {'电价':>6} │ {'T预报':>6} │"
          f" {'冷却':>4} │ {'功率kW':>6} │ {'电费':>6}")
    print("  " + "-" * 80)
    for k in range(len(schedule)):
        abs_dt = ws + datetime.timedelta(hours=k)
        tag = config.get_tariff_label(abs_dt.hour)
        print(f"  {k:>5} {abs_dt.strftime('%m-%d %H:%M'):>11} │"
              f" {ph[k]:>6.3f} │ {T_fc[k + 1]:>6.1f} │"
              f" {'开' if schedule[k] else '关':>4} │"
              f" {P_cool[k] / 1000:>6.1f} │ {cost_h[k]:>6.2f} ({tag})")
    print("=" * 88)
    print(f"  窗口计划总电费: {opt['total_cost']:.2f} 元 │"
          f" 冷却 {int(np.sum(schedule))} h │"
          f" 求解 {res['solve_time_s']:.2f} s │"
          f" 覆写: {res['overrides'] if res['overrides'] else '无'}")


def save_to_db(res, meas, config):     
    """入库：cooling_rolling_runs + cooling_applied_hours。
    失败只告警，不阻塞控制。"""
    try:
        from .save_database import DB_RollingResult
        db = DB_RollingResult(config)
        db.create_rolling_tables()

        ws = res["window_start"]
        ts_str = ws.strftime("%Y-%m-%d %H:%M:%S+08")
        opt = res.get("opt_result")
        if opt is not None:
            db.save_run({
                "run_ts": ts_str,
                "location": LOCATION_CONFIG.get("location", "Shenzhen_HETAO"),
                "window_start": ts_str,
                "window_end": (ws + datetime.timedelta(hours=24))
                              .strftime("%Y-%m-%d %H:%M:%S+08"),
                "meas_T_ely": meas.T_ely,
                "meas_T_stack": meas.T_stack,
                "meas_soc": meas.SoC,
                "meas_charge_latch": meas.charge_latch,
                "prev_cool_action": meas.prev_cool_action,
                "fc_ambient": list(res["weather"]["temperature"]),
                "fc_irradiation": list(res["weather"]["radiation"]),
                "fc_price": list(res["price_hourly"]),
                "opt_schedule": [int(x) for x in opt["schedule"]],
                "opt_temp_forecast": list(opt["T_forecast"]),
                "opt_total_cost": float(opt["total_cost"]),
                "first_action": res["first_action"],
                "raw_first_action": res["raw_first_action"],
                "commit_overridden": "commit_hold" in res["overrides"],
                "safety_override": next(
                    (o for o in res["overrides"] if o.startswith("safety")), ""),
                "weather_source": res["weather"].get("source", ""),
                "price_tier": ",".join(
                    f"{d}:{t}" for d, t in sorted(res["price_tiers"].items())),
                "solve_time_s": res["solve_time_s"],
            })
        db.save_applied_hour({
            "hour_start": ts_str,
            "run_ts": ts_str,
            "action": res["first_action"],
            "action_source": (res["overrides"][0] if res["overrides"]
                              else "new_plan"),
            "meas_T_before": meas.T_ely,
            "meas_T_after": None,
            "fc_temp": (float(res["weather"]["temperature"][0])
                        if res["weather"] is not None else None),
            "truth_ambient": None,
            "truth_price": None,
            "step_cost": None,
        })
        db.close()
        print("  ✅ 已入库 cooling_rolling_runs / cooling_applied_hours")
    except Exception as e:
        print(f"  ⚠️  入库失败（跳过，不阻塞控制）: {e}")

def start_power_price_collector(wait=False):
      """以独立子进程启动电价采集脚本（data_collection/power_price.py）。

      非阻塞：调度主流程不等待采集完成（电价有三级回退兜底，采集失败不影响调度）。
      返回 subprocess.Popen 对象（启动失败返回 None）。
      """
      script = os.path.join(
          os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # → 脚本\ 目录
          "data_collection", "power_price.py")
      if not os.path.exists(script):
          print(f"  ⚠️  电价采集脚本不存在: {script}（跳过，电价走DB回退链）")
          return None

      out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
      os.makedirs(out_dir, exist_ok=True)
      log_path = os.path.join(out_dir, "power_price.log")
      log_f = open(log_path, "a", encoding="utf-8")

      proc = subprocess.Popen(
          [sys.executable, "-X", "utf8", "-u", script],   # -X utf8: 子进程UTF-8输出,emoji不再触发GBK编码崩溃; -u 日志实时落盘
          stdout=log_f, stderr=subprocess.STDOUT,
          stdin=subprocess.DEVNULL,             # 关键：让结尾的 input() 立即 EOF，不挂起
          cwd=os.path.dirname(script),          # 工作目录=脚本所在目录（截图/配置路径依赖它）
          creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
      )
      print(f"  ✅ 电价采集子进程已启动 PID={proc.pid}")
      print(f"     日志: {log_path}")
      return proc


def main():
    import time
    t1=time.time()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="VRFB 冷却滚动时域调度入口")
    # 单步模式
    parser.add_argument("--once", metavar="DATETIME", nargs="?",
                        const="now", default=None,
                        help="单步模式：'YYYY-MM-DD HH:MM' 或不带值=当前整点")
    # parser.add_argument("--t-current", type=float, default=28.0,
    #                     help="当前电解液温度 [°C]（默认 28.0）")
    parser.add_argument("--soc", type=float, default=0.5,
                        help="当前 SoC [0-1]（默认 0.5）")
    parser.add_argument("--prev-action", type=int, default=0,
                        help="上一小时冷却动作 0/1")
    parser.add_argument("--latch", type=int, default=0,
                        help="当前 BMS 充电锁存 0/1")
    parser.add_argument("--no-db", action="store_true", help="不入库")

    # 真实数据持续滚动模式
    parser.add_argument("--live", action="store_true",
                        help="真实数据·持续滚动（真天气+真电价+你提供的24h温度）")

    parser.add_argument("--realtime", action="store_true",   
                        help="--live 用：真·实时模式（每整点一轮；默认快速演播）")
    parser.add_argument("--lead-minutes", type=int, default=5,
                        help="--live" \
                        " 用：提前几分钟算好并备好 0/1 指令给 PLC"
                             "（默认 5；0=整点才算。控制格仍对齐整点，只是提前计算）")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[2026, 2027])
    args = parser.parse_args()


    # ---------------- 真实数据·持续滚动模式 ----------------
    if args.live:

        from .live_runner import run_live
        start = None
        if args.once not in (None, "now"):
            start = datetime.datetime.strptime(args.once, "%Y-%m-%d %H:%M")

        # 指定日期 → 离线滚动（回测/预排）
        if start is not None:
            realtime, lead = False, 0
        else:
            realtime, lead = args.realtime, args.lead_minutes

        run_live(start_dt=start,
                 realtime=realtime,
                   initial_soc=args.soc,
                 
                 prev_action=args.prev_action, latch=args.latch,
                 lead_minutes=lead)
        return

def daily_price_collector_loop():
    #每天00：00触发一次电价采集
    now0=datetime.datetime.now()
    last_fire_date=now0.date() if now0.hour==0 else None    #0点归档，启动采集电价已经覆盖今天
    proc=None

    while True:
        now=datetime.datetime.now()
        next_mid=(now+datetime.timedelta(days=1)).replace( 
            hour=0,minute=0,second=0,microsecond=0)
        remain=(next_mid-now).total_seconds()
        time.sleep(30 if remain <=0 else min(remain,30))   #30s轮询，抗时钟跳变
        now=datetime.datetime.now()
        if now.hour==0 and now.date()!=last_fire_date:  #进入0点今天还没采集
            last_fire_date=now.date()
            if proc is not None and proc.poll() is None:  #上次采集还没跑完
                print(" ⚠️  上次采集子进程 PID={proc.pid} 仍在运行，跳过本次00:00触发")
                continue
            try:
                print(f"  ⏰ [{now:%Y-%m-%d %H:%M:%S}] 每日定时电价采集（00:00）")
                proc=start_power_price_collector()
            except Exception as e:
                print(f"  ⚠️  定时采集启动失败（不阻塞调度）: {e}")
            
if __name__ == "__main__":
    price_proc = start_power_price_collector()  # 首次启动：立即采集

    threading.Thread(target=daily_price_collector_loop,name='daily_price-collector',daemon=True).start()
    time.sleep(60)
    main()
