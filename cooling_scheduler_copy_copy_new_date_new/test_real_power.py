# -*- coding: utf-8 -*-
"""
测试脚本：从 mqtt 数据库 BMS 表读取 7 个功率测点
  - PCS Real-Time Power
  - DCDC1/2/3 Charging Power
  - DCDC1/2/3 Discharging Power
参照 save_database.py 中 Real_time_db.get_real_power 的查询方式：
  1) 最近 window_minutes 的均值（模拟实时调度取值）
  2) 最近 24h 的原始时序 + 逐小时均值，保存 Excel
运行:  python test_real_power.py
"""
import datetime
import psycopg2
import pandas as pd

CN_TZ = datetime.timezone(datetime.timedelta(hours=8))

DB_CFG = dict(host='47.107.118.226', port=5432, dbname='mqtt',
              user='FrankS', password='Yuanji!!')

TABLE = 'BMS'
POWER_COLS = [
    'PCS Real-Time Power',
    'DCDC1 Charging Power', 'DCDC2 Charging Power', 'DCDC3 Charging Power',
    'DCDC1 Discharging Power', 'DCDC2 Discharging Power', 'DCDC3 Discharging Power',
]

OUT_XLSX = r'C:\Users\user\Desktop\work_documents\脚本\cooling_scheduler_copy_copy_new_date_new\real_power_test.xlsx'


def get_power_avg(conn, as_of=None, window_minutes=5, fallback=None):
    """与 get_real_power 相同的方式：取 as_of 前 window_minutes 内各测点均值。"""
    if as_of is None:
        as_of = datetime.datetime.now(CN_TZ)
    elif as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=CN_TZ)
    start = as_of - datetime.timedelta(minutes=window_minutes)
    agg = ','.join('avg("{}")'.format(c) for c in POWER_COLS)
    sql = 'SELECT {} FROM "{}" WHERE "timestamp" >= %s AND "timestamp" <= %s'.format(agg, TABLE)
    cur = conn.cursor()
    cur.execute(sql, (start, as_of))
    row = cur.fetchone()
    cur.close()
    if row is None:
        return {c: fallback for c in POWER_COLS}
    return {c: (float(v) if v is not None else fallback) for c, v in zip(POWER_COLS, row)}


def get_power_series(conn, hours=24):
    """拉取最近 hours 小时的原始时序。"""
    end = datetime.datetime.now(CN_TZ)
    start = end - datetime.timedelta(hours=hours)
    cols = ','.join('"{}"'.format(c) for c in POWER_COLS)
    sql = ('SELECT "timestamp", {} FROM "{}" '
           'WHERE "timestamp" >= %s AND "timestamp" <= %s '
           'ORDER BY "timestamp"').format(cols, TABLE)
    df = pd.read_sql(sql, conn, params=(start, end), parse_dates=['timestamp'])
    # Excel 不支持带时区的 datetime：统一转北京时间后去掉 tzinfo
    if not df.empty and df['timestamp'].dt.tz is not None:
        df['timestamp'] = (df['timestamp'].dt.tz_convert(CN_TZ)
                           .dt.tz_localize(None))
    return df


def main():
    conn = psycopg2.connect(**DB_CFG)
    conn.autocommit = True
    try:
        now = datetime.datetime.now(CN_TZ)
        print('=' * 70)
        print(f'当前时间: {now:%Y-%m-%d %H:%M:%S} (UTC+8)')

        # ---- 1) 实时均值(与 get_real_power 同款, 5 分钟窗口) ----
        print('\n[1] 最近 5 分钟均值 (实时调度取值方式):')
        avg5 = get_power_avg(conn, as_of=now, window_minutes=5)
        for k, v in avg5.items():
            print(f'    {k:28s} = {v}')

        # ---- 2) 最近 24h 原始时序 ----
        df = get_power_series(conn, hours=24)
        print(f'\n[2] 最近 24h 原始时序: {len(df)} 行')
        if df.empty:
            print('    !! 24h 窗口无数据, 改拉全量')
            df = get_power_series(conn, hours=24 * 5)
            print(f'    全量: {len(df)} 行, {df["timestamp"].min()} ~ {df["timestamp"].max()}')

        # ---- 3) 逐小时均值 ----
        df_h = df.set_index('timestamp')
        hourly = df_h.resample('1h').mean().round(2)
        print(f'\n[3] 逐小时均值 ({len(hourly)} 小时):')
        print(hourly.to_string())

        # ---- 4) 统计摘要 ----
        print('\n[4] 统计摘要 (原始时序):')
        print(df_h.describe().round(2).to_string())

        # ---- 5) 保存 Excel ----
        with pd.ExcelWriter(OUT_XLSX, engine='openpyxl') as w:
            df.to_excel(w, sheet_name='原始时序', index=False)
            hourly.reset_index().to_excel(w, sheet_name='逐小时均值', index=False)
            pd.DataFrame({'测点': list(avg5.keys()),
                          '最近5分钟均值': list(avg5.values())}).to_excel(
                w, sheet_name='实时均值', index=False)
            df_h.describe().round(2).to_excel(w, sheet_name='统计摘要')
        print(f'\n[5] 已保存 Excel: {OUT_XLSX}')
    finally:
        conn.close()


if __name__ == '__main__':
    main()
