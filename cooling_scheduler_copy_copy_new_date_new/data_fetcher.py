# ============================================================================
# cooling_scheduler/data_fetcher.py
# 模块 2：数据获取（天气 + 电价）
#
# 职责：
#   1. 从 Open-Meteo API 获取天气预报（逐时温度 + 辐照度）
#   2. 从 PostgreSQL 获取分时电价数据
#   3. 统一输出为逐时数组，供后续模块使用
#
# 依赖：模块 1（config）
# 被依赖：模块 4（基线规则）、模块 5（DP 优化器）、模块 10（主程序）
# ============================================================================

import numpy as np
import requests
import datetime
import sys

from .config import (
    SystemConfig, DB_CONFIG, LOCATION_CONFIG, WEATHER_API_CONFIG,CN_TZ
)


# 默认深圳 TOU 电价对 (谷, 峰) [元/kWh] —— 三级回退的最后一级
DEFAULT_PRICE_PAIR = (0.23, 1.11)

# 最近一次成功获取的天气窗口缓存（滚动运行 fail-operational 用）
_LAST_GOOD_WEATHER_WINDOW = None


# ============================================================================
# 2.1 天气数据获取
# ============================================================================

def fetch_weather_data(target_date=None, source="auto"):
    """
    获取目标日期的逐时天气数据。

    参数:
        target_date: 目标日期，格式 "YYYY-MM-DD"，默认今天
        source:      数据源
                     "auto"   → 根据日期自动判断（过去用历史API，未来用预报API）
                     "api"    → 强制使用 API
                     "test"   → 使用内置测试数据（深圳夏季典型日）

    返回:
        weather_hourly: 字典
            {
                "temperature": ndarray[24],  # 逐时环境温度 [°C]
                "radiation":   ndarray[24],  # 逐时短波辐照度 GHI [W/m²]
                "source":      str,          # 数据来源标签
                "date":        str,          # 日期
            }
    """
    if target_date is None:
        target_date = datetime.date.today().strftime("%Y-%m-%d")


    today = datetime.date.today().strftime("%Y-%m-%d")

    # 自动判断：目标日期在今天之前 → 历史API，今天及之后 → 预报API
    if source == "auto":
        if target_date < today:
            time_type = "history"
        else:
            time_type = "real"
    elif source == "api":
        time_type = "history" if target_date < today else "real"
    else:
        raise ValueError(f"未知的 source: {source}")

    lat = LOCATION_CONFIG["latitude"]
    lon = LOCATION_CONFIG["longitude"]
    tz = WEATHER_API_CONFIG["timezone"]
    hourly_vars = ",".join(WEATHER_API_CONFIG["hourly_vars"])
    models = WEATHER_API_CONFIG["models"]

    params = {
        "latitude":  lat,
        "longitude": lon,
        "hourly":    hourly_vars,
        "timezone":  tz,
        "models":    models,
    }

    if time_type == "history":
        url = "https://archive-api.open-meteo.com/v1/archive"
        params["start_date"] = target_date
        params["end_date"] = target_date
    else:
        url = "https://api.open-meteo.com/v1/forecast"
        params["forecast_days"] = WEATHER_API_CONFIG["forecast_days"]
    print("天气参数为：",url,params)
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # print(data)
        hourly = data.get("hourly", {})
        temps = hourly.get("temperature_2m", [])
        rads = hourly.get("shortwave_radiation", [])

        if not temps or not rads:
            print(f"  ⚠️  API 返回数据为空")
            

        temps_arr = np.array(temps, dtype=float)
        rads_arr = np.array(rads, dtype=float)

        # 截取 24 小时
        temps_24 = temps_arr[:24]
        rads_24 = rads_arr[:24]

        # 如果不足 24 小时，用最后一个值填充
        if len(temps_24) < 24:
            pad = 24 - len(temps_24)
            temps_24 = np.pad(temps_24, (0, pad),
                              mode="constant",
                              constant_values=temps_24[-1])
            rads_24 = np.pad(rads_24, (0, pad),
                             mode="constant",
                             constant_values=0.0)

        # 处理缺失值
        temps_24 = np.where(np.isnan(temps_24),
                            np.nanmean(temps_24), temps_24)
        rads_24 = np.where(np.isnan(rads_24), 0.0, rads_24)

        source_label = "历史API" if time_type == "history" else "预报API"

        return {
            "temperature": temps_24,
            # "radiation" 为下游模块（optimizer/baseline/rolling_runner/
            # output_formatter/save_database）统一使用的键名；
            # 数据来源为 API 的 shortwave_radiation（水平面总短波辐照 GHI）
            "radiation":   rads_24,
            "source":      source_label,
            "date":        target_date,
        }

    except Exception as e:
        print(f"  ⚠️  API 请求失败: {e}")
        return  None


# ============================================================================
# 2.1b 滑动窗口天气获取（滚动时域用）
# ============================================================================

def _cn_today():
      """当前北京时间（naive）。全链路统一的时区基准，不依赖宿主机系统时区。"""
      return datetime.datetime.now(CN_TZ).replace(tzinfo=None)

def fetch_weather_window(start_dt, horizon_h=24):
    """
    获取滑动窗口 [start_dt, start_dt+horizon_h) 的逐时天气。

    参数:
        start_dt:   窗口起始时刻（datetime，本地 Asia/Shanghai 时间）
        horizon_h:  窗口长度 [小时]（默认 24）

    返回:
        weather_window: 字典
            {
                "temperature":  ndarray[horizon_h],  # 逐时环境温度 [°C]
                "radiation":    ndarray[horizon_h],  # 逐时短波辐照度 GHI [W/m²]
                "source":       "预报API" / "缓存",
                "window_start": str,                 # 窗口起点 ISO 时刻
            }
        获取失败且无缓存时返回 None（调用方走 fail-operational）。

    实现要点：
      - forecast_days=2 拉 48 个逐时点（从今天 00:00 起），覆盖跨午夜窗口；
      - 按 hourly.time 时间戳定位第一个 >= start_dt 的下标 i0，
        切 [i0:i0+horizon_h]，不做裸下标算术；
      - 不足 horizon_h 点：温度末值 pad、辐照 pad 0、NaN 沿用旧逻辑；
      - API 失败回退最近成功缓存（source="缓存"），无缓存返回 None。
    """
    global _LAST_GOOD_WEATHER_WINDOW

    lat = LOCATION_CONFIG["latitude"]
    lon = LOCATION_CONFIG["longitude"]
    tz = WEATHER_API_CONFIG["timezone"]
    hourly_vars = ",".join(WEATHER_API_CONFIG["hourly_vars"])
    models = WEATHER_API_CONFIG["models"]

    # ---- 日期感知：过去→archive历史API；现在/未来→forecast预报API ----
    # now = datetime.datetime.now()
    now=_cn_today()
    is_past = (start_dt+datetime.timedelta(hours=horizon_h)) <now
    params = {
        "latitude":      lat,
        "longitude":     lon,
        "hourly":        hourly_vars,
        "timezone":      tz,
        "models":        models,
    }
    if is_past:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params["start_date"] = start_dt.strftime("%Y-%m-%d")
        params["end_date"] = (start_dt + datetime.timedelta(hours=horizon_h)).strftime("%Y-%m-%d")
        source_label = "历史API"
    else:
        url = "https://api.open-meteo.com/v1/forecast"
        # params["forecast_days"] = 2          # 48h 覆盖，支持跨午夜窗口
        # ---- 加固③：按窗口实际跨度动态取天数，不再写死 2 ----
        # forecast_days = "从 API 时区的今天起算 N 天"，返回 N×24 个点。
        # 需要的天数 = 窗口末点落在第几天。realtime 模式恒为 2；
        # 快速演播模式墙钟不动而 start_dt 前推，跨午夜后会需要 3 天，
        # 写死 2 会导致尾部缺点并被静默 flat-pad（回归测试数据失真）。
        last_hour = start_dt + datetime.timedelta(hours=horizon_h - 1)
        days_needed = (last_hour.date() - _cn_today().date()).days + 1
        params["forecast_days"] = int(max(1, min(days_needed, 16)))  # API 上限 16 天
        if days_needed>16:
            print(f"窗口需要{days_needed} 天预报，超API上限16天，尾部将被补齐")
        source_label = "预报API"

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        rads = hourly.get("shortwave_radiation", [])
        if not times or not temps or not rads:
            raise ValueError("API 返回数据为空")

        # 按时间戳定位窗口起点
        start_str = start_dt.strftime("%Y-%m-%dT%H:00")
        i0 = None
        for i, t in enumerate(times):
            if t >= start_str:
                i0 = i
                break
        if i0 is None:
            raise ValueError(f"无 >= {start_str} 的预报点")
        
        # ---- 加固①：起点必须精确命中，否则说明序列与请求时刻错位 ----
        # 触发场景：宿主机时区≠Asia/Shanghai、API 返回范围变化、夏令时切换日。
        # 旧逻辑只用 ">=" 取第一个，错位时会静默整体平移窗口（拿到别的小时）。
        aligned = (times[i0] == start_str)
        if not aligned:
            print(f"  ⚠️   预报起点未对齐: 请求 {start_str}，实得首点 {times[i0]}"
                f"（序列 {times[0]}~{times[-1]}）→ 窗口已整体偏移，请检查宿主机时区")
            
        temps_arr = np.array(temps[i0:i0 + horizon_h], dtype=float)
        rads_arr = np.array(rads[i0:i0 + horizon_h], dtype=float)

        # 不足 horizon_h 点：温度末值 pad、辐照 pad 0
        n_pad = horizon_h - len(temps_arr)
        if n_pad > 0:
              # ---- 加固②：补齐必须出声，且写入返回值供入库/复盘追溯 ----
            last_t = float(temps_arr[-1]) if len(temps_arr) else 25.0
            print(f"  ⚠️   预报点不足 {horizon_h}（实得 {len(temps_arr)}，缺 {n_pad}）"
                f"→ 温度按末值 {last_t:.1f}°C 平推、辐照补 0")
            temps_arr = np.pad(temps_arr, (0, n_pad), mode="constant",
                                constant_values=last_t)
            rads_arr = np.pad(rads_arr, (0, n_pad), mode="constant",
                            constant_values=0.0)

        # 处理缺失值
        temps_arr = np.where(np.isnan(temps_arr),
                             np.nanmean(temps_arr), temps_arr)
        rads_arr = np.where(np.isnan(rads_arr), 0.0, rads_arr)

        result = {
            "temperature":  temps_arr,
            "radiation":    rads_arr,
            "source":       source_label,
            "window_start": start_str,
            "n_padded":max(0,n_pad),
            "aligned":aligned,

        }
        _LAST_GOOD_WEATHER_WINDOW = result
        return result

    except Exception as e:
        print(f"  ⚠️  天气窗口获取失败: {e}")
        if _LAST_GOOD_WEATHER_WINDOW is not None:
            print("  ↩ 使用缓存天气窗口")
            cached = dict(_LAST_GOOD_WEATHER_WINDOW)
            cached["source"] = "缓存"
            return cached
        return None


# ============================================================================
# 2.1c 逐日电价对获取与三级回退（滚动时域用）
# ============================================================================

def fetch_price_pair(target_date):
    """
    查询指定日历日已发布的 (谷, 峰) 电价对 [元/kWh]。

    参数:
        target_date: 日期字符串 "YYYY-MM-DD"

    返回:
        [valley, peak]（成功）或 None（失败/未发布）。
        永不返回 24 元素数组。
    """
    try:
        import psycopg2
        import pandas as pd
    except ImportError:
        print("  ⚠️  psycopg2 未安装，无法查询数据库电价")
        return None

    try:
        conn = psycopg2.connect(
            host=DB_CONFIG["ts_host"],
            port=DB_CONFIG["ts_port"],
            dbname=DB_CONFIG["ts_dbname"],
            user=DB_CONFIG["ts_user"],
            password=DB_CONFIG["ts_password"],
        )
        conn.autocommit = True
        sql = f"""
            SELECT time, city, price_type, price_cny_mwh
            FROM {DB_CONFIG["measurement"]}
            WHERE city = %s
            AND time >= %s::timestamptz
            AND time <= %s::timestamptz
            ORDER BY time ASC
        """
        start_time = f"{target_date} 00:00:00+08"
        end_time = f"{target_date} 23:59:59+08"
        df = pd.read_sql(sql, conn, params=("Shenzhen", start_time, end_time))
        conn.close()

        if df.empty:
            return None
        df["price_cny_mwh"] = df["price_cny_mwh"].values / 1000.0
        valley_rows = df[df["price_type"] == "valley"]["price_cny_mwh"].values
        peak_rows = df[df["price_type"] == "peak"]["price_cny_mwh"].values
        if len(valley_rows) == 0 or len(peak_rows) == 0:
            return None
        return [float(valley_rows[0]), float(peak_rows[0])]

    except Exception as e:
        print(f"  ⚠️  电价查询失败 ({target_date}): {e}")
        return None


def resolve_price_pairs(dates, today):
    """
    为多个日历日做电价三级回退。

    每个日期 d：
      ① 查 d 的 DB 电价对；
      ② 失败且 d != today → 用 today 的电价对代替；
      ③ 仍失败 → DEFAULT_PRICE_PAIR。

    参数:
        dates: 日期字符串列表（窗口覆盖的日历日，如 [今天, 明天]）
        today: 今天的日期字符串（回退参照日）

    返回:
        pairs: {date_str: [valley, peak]}
        tiers: {date_str: "db" | "today_fallback" | "default"}
    """
    pairs, tiers = {}, {}
    today_pair = None
    for d in dates:
        pair = fetch_price_pair(d)
        if pair is not None:
            pairs[d] = pair
            tiers[d] = "db"
            if d == today:
                today_pair = pair
            continue
        # 回退 ②：用今天的电价对
        if d != today:
            if today_pair is None:
                today_pair = fetch_price_pair(today)
            if today_pair is not None:
                pairs[d] = list(today_pair)
                tiers[d] = "today_fallback"
                continue
        # 回退 ③：默认 TOU
        pairs[d] = list(DEFAULT_PRICE_PAIR)
        tiers[d] = "default"
    return pairs, tiers


def build_price_hourly(window_start_dt, horizon_h, pair_by_date, config):
    """
    拼接滑动窗口的逐时电价（可跨午夜）。

    每小时：按其日历日选 (谷, 峰) 价格对，按钟表时刻在
    config.price["windows"]（钟表绑定、跨日共用）选档；
    flat/critical 按该日价格对现算（单个 SystemConfig 装不下两天
    的价格，不走 config.price 展开）。

    参数:
        window_start_dt: 窗口起始时刻（datetime）
        horizon_h:       窗口长度 [小时]
        pair_by_date:    {date_str: [valley, peak]}（resolve_price_pairs 输出）
        config:          SystemConfig 实例（只用其 price["windows"] 时段表）

    返回:
        ndarray[horizon_h] 逐时电价 [元/kWh]
    """
    prices = np.zeros(horizon_h)
    for k in range(horizon_h):
        abs_dt = window_start_dt + datetime.timedelta(hours=k)
        d_str = abs_dt.strftime("%Y-%m-%d")
        valley, peak = pair_by_date[d_str]
        flat = peak * 0.47 + valley * 0.53
        critical = peak * 1.25
        levels = {1: valley, 2: flat, 3: peak, 4: critical}
        p = flat                      # 窗外默认：该日平时价
        for row in config.price["windows"]:
            h0, h1, lid = row[0], row[1], int(row[2])
            if h0 <= abs_dt.hour < h1:
                p = levels.get(lid, flat)
                break
        prices[k] = p
    return prices


# ============================================================================
# 2.2 电价数据获取
# ============================================================================

def fetch_price_data(target_date=None, source="auto", config=None):
    """
    获取目标日期的逐时电价数据。

    参数:
        target_date: 目标日期，格式 "YYYY-MM-DD"，默认今天
        source:      数据源
                     "auto" → 先尝试数据库，失败则用默认时段表
                     "db"   → 强制使用数据库
                     "test" → 使用默认时段表（深圳电价）
        config:      SystemConfig 实例（用于获取默认电价和数据库配置）

    返回:
        price_hourly: ndarray[24]，逐时电价 [元/kWh]
    """
    if target_date is None:
        target_date = datetime.date.today().strftime("%Y-%m-%d")

    # test 源：直接返回默认电价对（离线测试用）
    if source == "test":
        return list(DEFAULT_PRICE_PAIR)

    # 延迟创建 config，只在需要 fallback 时创建
    # 避免循环依赖：fetch_all_data -> fetch_price_data -> SystemConfig

    # ----- 数据库模式 -----
    try:
        import psycopg2
        import pandas as pd
        today = datetime.date.today().strftime("%Y-%m-%d")
        # 未来日期电价预测钩子：暂未实现，显式返回 None
        # （滚动链路经 resolve_price_pairs 三级回退；日前链路落默认 TOU）
        # 将来扩展：输入天气特征预测明日谷/峰电价
        #   predict_valley_price = valley_model.predict(weather_features)
        #   predict_peak_price   = peak_model.predict(weather_features)
        if target_date > today:
            return None
        else:
            conn = psycopg2.connect(
                host=DB_CONFIG["ts_host"],
                port=DB_CONFIG["ts_port"],
                dbname=DB_CONFIG["ts_dbname"],
                user=DB_CONFIG["ts_user"],
                password=DB_CONFIG["ts_password"],
            )
            conn.autocommit = True

            sql = f"""
                SELECT time, city, price_type, price_cny_mwh
                FROM {DB_CONFIG["measurement"]}
                WHERE city = %s
                AND time >= %s::timestamptz
                AND time <= %s::timestamptz
                ORDER BY time ASC
            """
            start_time = f"{target_date} 00:00:00+08"
            end_time = f"{target_date} 23:59:59+08"

            df = pd.read_sql(sql, conn,
                            params=("Shenzhen", start_time, end_time))
            if not df.empty and hasattr(df["time"].dtype, "tz") and df["time"].dt.tz is not None: 
                df["time"] = df["time"].dt.tz_convert("Asia/Shanghai")

            conn.close()

            if df.empty:
                print(f"  ⚠️  数据库查询结果为空，使用默认电价对")
                return list(DEFAULT_PRICE_PAIR)

            # 转换为单位/1000
            df["price_cny_mwh"] = df["price_cny_mwh"].values / 1000.0
            print("转换为元/kWh",df)
            price_kwh=[df[df['price_type']=='valley']['price_cny_mwh'].values[0],df[df['price_type']=='peak']['price_cny_mwh'].values[0]] 
            print("数据库的电价为:",price_kwh)
        
        return price_kwh

    except ImportError:
        print(f"  ⚠️  psycopg2 未安装，使用默认电价对")
        return list(DEFAULT_PRICE_PAIR)

    except Exception as e:
        print(f"  ⚠️  数据库连接失败: {e}")
        if source == "db":
            raise
        return list(DEFAULT_PRICE_PAIR)
    
def fetch_all_data(target_date=None):
      """
      统一获取所有数据（天气 + 电价 + 配置）
   
      返回:
          config: SystemConfig 实例（已配置实际电价）
          weather: 天气数据字典
          price_hourly: 24小时电价数组
      """
      if target_date is None:
          target_date = datetime.date.today().strftime("%Y-%m-%d")

      # 1. 获取原始电价数据
      price_kwh = fetch_price_data(target_date)

      # 2. 创建配置（使用实际电价）
      if price_kwh is not None and len(price_kwh) >= 2:
          # price_kwh[0] = 谷时, price_kwh[1] = 峰时
          price_list=price_kwh
          print(f"  使用数据库电价: 谷时={price_list[0]:.4f}, 峰时={price_list[1]:.4f} 元/kWh")
          config = SystemConfig(price_list=price_list)
      else:
          print(f"  使用默认电价")
          config = SystemConfig(price_list=list(DEFAULT_PRICE_PAIR))

      # 3. 获取天气数据
      weather = fetch_weather_data(target_date, source="auto")

      # 4. 生成24小时电价序列
      price_hourly = np.zeros(24)
      for h in range(24):
          price, _ = config.get_price_and_tariff(h)
          price_hourly[h] = price

      return config, weather, price_hourly


# ============================================================================
# 2.3 数据打印（调试用）
# ============================================================================

def print_weather_summary(weather):
    """打印天气数据摘要"""
    print(f"\n--- 天气数据摘要 ---")
    print(f"  日期:   {weather['date']}")
    print(f"  来源:   {weather['source']}")
    print(f"  温度:   {np.min(weather['temperature']):.1f} ~ "
          f"{np.max(weather['temperature']):.1f} °C")
    print(f"  辐照度: {np.max(weather['radiation']):.0f} W/m² 峰值")
    print(f"  逐时温度: {np.array2string(weather['temperature'], precision=1, separator=', ')}")
    print(f"  逐时辐照: {np.array2string(weather['radiation'], precision=0, separator=', ')}")


def print_price_summary(price_hourly, config=None):
    """打印电价数据摘要"""
    if config is None:
        config = SystemConfig()

    print(f"\n--- 电价数据摘要 ---")
    print(f"  谷时: {config.price['valley']:.4f} 元/kWh")
    print(f"  平时: {config.price['flat']:.4f} 元/kWh")
    print(f"  峰时: {config.price['peak']:.4f} 元/kWh")
    print(f"  尖峰: {config.price['critical']:.4f} 元/kWh")
    print(f"  逐时电价: {np.array2string(price_hourly, precision=3, separator=', ')}")

    # 统计各时段（根据 config 中的 TOU 时段表）
    valley_h = 0
    flat_h = 0
    peak_h = 0
    critical_h = 0
    for h in range(24):
        label = config.get_tariff_label(h)
        if label == "谷":
            valley_h += 1
        elif label == "平":
            flat_h += 1
        elif label == "峰":
            peak_h += 1
        elif label == "尖峰":
            critical_h += 1
    print(f"  谷时 {valley_h}h / 平时 {flat_h}h / 峰时 {peak_h}h / 尖峰 {critical_h}h")
