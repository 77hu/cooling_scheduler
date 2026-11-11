import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import requests
import datetime
import sys
import psycopg2
from psycopg2.extras import execute_values
from .config import SystemConfig, LOCATION_CONFIG,WEATHER_API_CONFIG,CN_TZ
from collections import defaultdict
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

from .data_fetcher import fetch_all_data
from .config import SystemConfig



# ============================================================================
# 滚动时域结果存储（v2）
# ============================================================================

class DB_RollingResult:
    """
    小时级滚动调度结果数据库存储类。

    两张表：
      cooling_rolling_runs   —— 每次 run_once 一行（run_ts 主键，
                                ON CONFLICT DO NOTHING：重跑不覆盖历史）
      cooling_applied_hours  —— 每执行小时一行（hour_start 主键，
                                ON CONFLICT DO UPDATE：便于事后补真值）

    DB 写失败只告警、不阻塞控制（调用方 try/except 包裹）。
    """

    RUNS_TABLE = "cooling_rolling_runs"
    HOURS_TABLE = "cooling_applied_hours"

    def __init__(self, config):
        self.CONFIG = config
        self.connect_database(config)

    def connect_database(self, cfg):
        try:
            conn = psycopg2.connect(
                host=cfg['ts_host'],
                port=cfg['ts_port'],
                dbname=cfg['ts_dbname'],
                user=cfg['ts_user'],
                password=cfg['ts_password'],
                # TCP keepalive: 防长驻进程空闲连接被NAT/防火墙/服务端静默掐断
                keepalives=1, keepalives_idle=30,
                keepalives_interval=10, keepalives_count=3,
            )
            conn.autocommit = True
            self.conn = conn
            return self.conn
        except psycopg2.Error as exc:
            print(f"数据库连接失败: {exc}")
            raise

    def _ensure_conn(self):
        """连接失效(被服务端关闭/空闲超时)时自动重连。"""
        if getattr(self, "conn", None) is None or self.conn.closed:
            self.connect_database(self.CONFIG)

    def _execute(self, sql, params=None):
        """执行写入语句；遇连接被服务端关闭则重连重试一次。"""
        for _ in range(2):
            try:
                self._ensure_conn()
                cur = self.conn.cursor()
                cur.execute(sql, params)
                cur.close()
                return
            except psycopg2.OperationalError as e:
                print(f"  ⚠ 结果库连接异常: {str(e).splitlines()[0]} → 重连重试")
                try:
                    self.conn.close()
                except Exception:
                    pass
                self.conn = None
        print("  ⚠ 结果库重连后仍失败, 本次写入放弃(不阻塞控制)")

    def create_rolling_tables(self):
        cur = self.conn.cursor()
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.RUNS_TABLE} (
                "run_ts"           TIMESTAMPTZ PRIMARY KEY,
                "location"         TEXT,
                "window_start"     TIMESTAMPTZ,
                "window_end"       TIMESTAMPTZ,
                "meas_T_ely"       REAL,
                "meas_T_stack"     REAL,
                "meas_soc"         REAL,
                "meas_charge_latch" SMALLINT,
                "prev_cool_action" SMALLINT,
                "fc_ambient"       REAL[],
                "fc_irradiation"   REAL[],
                "fc_price"         REAL[],
                "opt_schedule"     SMALLINT[],
                "opt_temp_forecast" REAL[],
                "opt_total_cost"   REAL,
                "first_action"     SMALLINT,
                "raw_first_action" SMALLINT,
                "commit_overridden" BOOLEAN,
                "safety_override"  TEXT,
                "weather_source"   TEXT,
                "price_tier"       TEXT,
                "solve_time_s"     REAL,
                "created_at"       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.HOURS_TABLE} (
                "hour_start"    TIMESTAMPTZ PRIMARY KEY,
                "run_ts"        TIMESTAMPTZ,
                "action"        SMALLINT,
                "action_source" TEXT,
                "meas_T_before" REAL,
                "meas_T_after"  REAL,
                "fc_temp"       REAL,
                "truth_ambient" REAL,
                "truth_price"   REAL,
                "step_cost"     REAL
            );
        """)
        cur.close()
        print(f"  表 {self.RUNS_TABLE} / {self.HOURS_TABLE} 创建成功/已存在")

    @staticmethod
    def _to_python_type(val):
        """递归将 numpy 类型转为原生 Python 类型（psycopg2 适配）。"""
        if val is None:
            return None
        if hasattr(val, 'item'):
            return val.item()
        if isinstance(val, list):
            return [DB_RollingResult._to_python_type(v) for v in val]
        if isinstance(val, tuple):
            return tuple(DB_RollingResult._to_python_type(v) for v in val)
        if isinstance(val, np.ndarray):
            return [DB_RollingResult._to_python_type(v) for v in val.tolist()]
        return val

    def save_run(self, rec):
        """写入一次 run_once 记录（run_ts 冲突时 DO NOTHING）。"""
        cols = ["run_ts", "location", "window_start", "window_end",
                "meas_T_ely", "meas_T_stack", "meas_soc",
                "meas_charge_latch", "prev_cool_action",
                "fc_ambient", "fc_irradiation", "fc_price",
                "opt_schedule", "opt_temp_forecast", "opt_total_cost",
                "first_action", "raw_first_action",
                "commit_overridden", "safety_override",
                "weather_source", "price_tier", "solve_time_s"]
        col_str = ", ".join(f'"{c}"' for c in cols)
        placeholders = ", ".join(["%s"] * len(cols))
        sql = (f"INSERT INTO {self.RUNS_TABLE} ({col_str}) "
               f"VALUES ({placeholders}) "
               f'ON CONFLICT ("run_ts") DO NOTHING')
        values = tuple(self._to_python_type(rec.get(c)) for c in cols)
        self._execute(sql, values)

    def save_applied_hour(self, rec):
        """写入一执行小时（hour_start 冲突时补更新真值列）。

        字段命名与 cooling_schedule_result 的逐时数组字段对齐（标量化）：
          price / ambient / irradiation / forecast_temp / status / power / cost_hourly
          —— 即 schedule_result 各数组在"本采纳小时"上的单值。
        """
        cols = ["hour_start", "run_ts", "action", "action_source",
                  "meas_T_before", "meas_T_after", "ambient",
                  "truth_ambient", "truth_price", "cost_hourly",
                  "price", "tariff_label", "raw_action", "forecast_temp",
                  "soc_before", "soc_after", "p_target_w", "power_source",
                  "cum_cost", "irradiation", "status", "power"]
        col_str = ", ".join(f'"{c}"' for c in cols)
        placeholders = ", ".join(["%s"] * len(cols))
        update_cols = ["meas_T_after", "ambient", "truth_ambient",
                       "truth_price", "cost_hourly"]
        update_str = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in update_cols)
        sql = (f"INSERT INTO {self.HOURS_TABLE} ({col_str}) "
               f"VALUES ({placeholders}) "
               f'ON CONFLICT ("hour_start") DO UPDATE SET {update_str}')
        values = tuple(self._to_python_type(rec.get(c)) for c in cols)
        self._execute(sql, values)

    def query_runs(self, start_ts, end_ts):
        sql = (f"SELECT * FROM {self.RUNS_TABLE} "
               f'WHERE "run_ts" >= %s AND "run_ts" < %s '
               f'ORDER BY "run_ts" ASC')
        return pd.read_sql(sql, self.conn, params=(start_ts, end_ts))

    def query_applied(self, start_ts, end_ts):
        sql = (f"SELECT * FROM {self.HOURS_TABLE} "
               f'WHERE "hour_start" >= %s AND "hour_start" < %s '
               f'ORDER BY "hour_start" ASC')
        return pd.read_sql(sql, self.conn, params=(start_ts, end_ts))

    def close(self):
        if self.conn and not self.conn.closed:
            self.conn.close()


class Real_time_db:

    ##"Pos Elect Tank Lower Temp"，"Neg Elect Tank Lower Temp"任意一个满足温度设定就可与开启冷却指令
    TEMP_COLS = [
          "Pos Elect Tank Upper Temp",   # 正极电解液罐上部温度
          "Pos Elect Tank Lower Temp",   # 正极电解液罐下部温度
        #   "Neg Elect Tank Upper Temp",   # 负极电解液罐上部温度
        #   "Neg Elect Tank Lower Temp",   # 负极电解液罐下部温度
        #   "Pos Return Pipe Upstr Temp",  # 正极回流管上游温度
        #   "Pos Return Pipe Dnstr Temp",  # 正极回流管下游温度
        #   "Neg Return Pipe Upstr Temp",  # 负极回流管上游温度
        #   "Neg Return Pipe Dnstr Temp",  # 负极回流管下游温度
        #   "Container Top Temperature",   # 集装箱顶部温度
        #   "Liquid Temperature",          # 液体温度
      ]
    def __init__(self,config=None):
        self.CONFIG=config if config is not None else SystemConfig()
        self.connect_database(self.CONFIG)

    def connect_database(self,cfg=None,database="mqtt"):
        try:
            self.conn=psycopg2.connect(
                host=cfg['ts_host'],
                port=cfg['ts_port'],
                dbname=database,
                user=cfg['ts_user'],
                password=cfg['ts_password'],
                # TCP keepalive: 防空闲连接被NAT/防火墙/服务端静默掐断
                keepalives=1, keepalives_idle=30,
                keepalives_interval=10, keepalives_count=3,
            )
            self.conn.autocommit=True
            return self.conn
        except psycopg2.Error as exc:
            print("数据库连接失败：{}".format(exc))
            raise

    def _ensure_conn(self):
        """连接失效(被服务端关闭/空闲超时)时自动重连。"""
        if getattr(self, "conn", None) is None or self.conn.closed:
            self.connect_database(self.CONFIG)

    def _fetchone(self, sql, params=()):
        """执行查询返回单行；遇连接被服务端关闭则重连重试一次。"""
        for _ in range(2):
            try:
                self._ensure_conn()
                cur = self.conn.cursor()
                cur.execute(sql, params)
                row = cur.fetchone()
                cur.close()
                return row
            except psycopg2.OperationalError as e:
                print(f"  ⚠ DB连接异常: {e} → 重连重试")
                try:
                    self.conn.close()
                except Exception:
                    pass
                self.conn = None
        return None

    def get_current_max_temp(self,as_of=None,window_minutes=5,table="BMS",fallback=28.0,return_name=None):

        """取 as_of（默认 now，实则应传 ready_at=整点前5分钟）前 window_minutes 的
      温度窗口，每个测点求均值后取最大，作为小时锚点温度。"""
        if as_of is None:                                   
              as_of = datetime.datetime.now(CN_TZ)
        elif as_of.tzinfo is None:
          as_of = as_of.replace(tzinfo=CN_TZ)
        start=as_of-datetime.timedelta(minutes=window_minutes)
        agg=','.join('avg("{}")'.format(c) for c in self.TEMP_COLS)
        sql=('SELECT {} FROM "{}" ' 'WHERE "timestamp"  >=%s AND "timestamp" <=%s').format(agg,table)
        row = self._fetchone(sql, (start, as_of))
        if row is None:
          return (fallback, None) if return_name else fallback
        pairs = [(name, float(v)) for name, v in zip(self.TEMP_COLS, row) if v is not None]
        if not pairs:
            return (fallback, None) if return_name else fallback

        name, val = max(pairs, key=lambda p: p[1]) 
        # vals_sorted=sorted(pairs,key=lambda p:p[1])
        # n=len(vals_sorted)
        # if n%2==1:
        #     name,val=vals_sorted[n//2]   ##取基数个的中位数
        # else:
        #     (name,v1),(_,v2)=vals_sorted[n//2-1],vals_sorted[n//2]  #取偶数个名字位置靠前的
        #     val=(v1+v2)/2.0
        # print(f"中位##{name}    中位数温度：：{np.round(val,2)}")

        print(f"最大## {name}  ##温度的是：{np.round(val,2)}")
        return (np.round(val,2), name) if return_name else val
    
    def get_real_power(self,as_of=None,table="BMS",fallback=0.0, max_stale_sec=60):
        if as_of is None:
            as_of=datetime.datetime.now(CN_TZ)
        elif as_of.tzinfo is None:
            as_of=as_of.replace(tzinfo=CN_TZ)
        
        POWER_COLS = [
          "PCS Real-Time Power",
          "Total Current",
        #   "DCDC1 Charging Power", "DCDC2 Charging Power", "DCDC3 Charging Power",
        #   "DCDC1 Discharging Power", "DCDC2 Discharging Power", "DCDC3 Discharging Power",
      ]
        cols=",".join('"{}"'.format(c) for c in POWER_COLS)
        sql=('SELECT "timestamp",{} FROM "{}" ' 'WHERE "timestamp" <=%s ' 'ORDER BY "timestamp" DESC LIMIT 1').format(cols,"BMS")

        row = self._fetchone(sql, (as_of,))
        # print(f"从数据库中拿到了数据：{row}")
        if row is None:
              return {"data_ts": None, "stale": True, "pcs_kw": 0.0, "p_target": 0.0,
                      "total_current": 0.0, "source": -1}
        ts=row[0]
        pcs_kw,pcs_current=[float(v) if v is not None else 0.0 for v in row[1:]]
        pcs_w=pcs_kw*1000.0           # PCS实时功率[W]
        
        EPS_A=2                         #电流
        EPS_W=1000.0              # 功率死区[W]       
        # if pcs_current>EPS:     #电流大于10，功率为负放电

        #     source =  1        
        # else:                   #电流小于10，功率为正充电
        
        #     source =  0       
        if abs(pcs_w) < EPS_W:
              source, p_target = -1, 0.0                 # 待机
        else:
            # 模型约定: 负=充电/正=放电, 与数据库相反 → 统一取反
            p_target = -pcs_w
            source = 1 if pcs_current > EPS_A else 0   # 1=放电 0=充电

        stale=(as_of-ts).total_seconds()>max_stale_sec
        info={"data_ts": ts, "stale": stale, "pcs_kw": pcs_kw,
                "p_target": p_target, "total_current": pcs_current, "source": source}
          
        return info


    def get_SoC(self,as_of=None,table="BMS",max_stale_sec=40,mismatch_pct=5.0):
        if as_of is None:
            as_of=datetime.datetime.now(CN_TZ)
        elif as_of.tzinfo is None:
            as_of=as_of.replace(tzinfo=CN_TZ)
        sql=('SELECT "timestamp","SOC_CPU Calculation","SOC_HMI Calculation"'
             'FROM "{}" WHERE "timestamp" <=%s '
             'ORDER BY "timestamp" DESC LIMIT 1').format(table)
        row = self._fetchone(sql, (as_of,))
        if row is None:
            return None,{'data_ts':None,"stale":True,"soc_cpu_pct":None,'soc_hmi_pct':None,
                         "mismatch":False,"invalid":True}
        ts,cpu,hmi=row
        cpu=float(cpu) if  cpu  is not None else None
        hmi=float(hmi) if hmi is not  None else None

        #临时使用数据。等待Soc_cpu恢复再切换回去
        cpu=hmi

        stale=(as_of-ts).total_seconds()>max_stale_sec
        invalid=(cpu is None) or not (0.0<=cpu<=100.0)
        mismatch=(cpu is not None and  hmi is not None and abs(cpu-hmi)>mismatch_pct)
        info={
            "data_ts":ts,'stale':stale,'soc_cpu_pct':cpu,'soc_hmi_pct':hmi,'mismatch':mismatch,
            "invalid":invalid
        }

        if invalid:
            return None,info
        return cpu/100.0,info    
    
    def close(self):
        """关闭数据库连接。"""
        if self.conn and not self.conn.closed:
            self.conn.close()


class Weather_data:

    WEATHER_TABLE = "weather_data"
    SCALAR_COLS = ["latitude", "longitude", "elevation", "generationtime_ms"]
    INT_ARRAY_COLS = {"cloud_cover", "relative_humidity_2m",
                        "wind_direction_10m", "weather_code"}
    def __init__(self,config=None):
            self.CONFIG=config if config is not None else SystemConfig()
            self.connect_database(self.CONFIG)
            
            self.hourly_vars = list(WEATHER_API_CONFIG["hourly_vars"])
            self.weather_table_cols = ["all_day"] + self.SCALAR_COLS + self.hourly_vars
    def connect_database(self,cfg=None,database="energy"):
            try:
                self.conn=psycopg2.connect(
                    host=cfg['ts_host'],
                    port=cfg['ts_port'],
                    dbname=database,
                    user=cfg['ts_user'],
                    password=cfg['ts_password'],
                    # TCP keepalive: 防空闲连接被NAT/防火墙/服务端静默掐断
                    keepalives=1, keepalives_idle=30,
                    keepalives_interval=10, keepalives_count=3,
                )
                self.conn.autocommit=True
                return self.conn
            except psycopg2.Error as exc:
                print("数据库连接失败：{}".format(exc))
                raise
    def _ensure_conn(self):
        """连接失效(被服务端关闭/空闲超时)时自动重连。"""
        if getattr(self, "conn", None) is None or self.conn.closed:
            self.connect_database(self.CONFIG, database="energy")
    def close(self):
        """关闭数据库连接。"""
        if getattr(self, "conn", None) and not self.conn.closed:
            self.conn.close()

    def get_weather_data(self,start_date="2026-01-01",end_date=None,past_days=2):
        params = {
                "latitude":  LOCATION_CONFIG["latitude"],
                "longitude": LOCATION_CONFIG["longitude"],
                "hourly": ",".join(WEATHER_API_CONFIG["hourly_vars"]),
                "timezone": WEATHER_API_CONFIG["timezone"],
                "models": WEATHER_API_CONFIG["models"],
                "past_days":past_days,
            }
        
        # if start_date < end_date :
        #     url = "https://archive-api.open-meteo.com/v1/archive"
        #     params["start_date"] = start_date
        #     params["end_date"] = end_date
            
        # else:
        url = "https://api.open-meteo.com/v1/forecast"
        params["forecast_days"] = WEATHER_API_CONFIG["forecast_days"]
            
    
        resp = requests.get(url, params=params, timeout=30)
        data = resp.json()
     
        # print(f"{data}")
        hourly = data["hourly"]
        # 1.解析全部时间，转为datetime
        time_list = pd.to_datetime(hourly["time"])
    
        # 2.按日期分组 key:date, value:该天所有下标
        day_index_map = defaultdict(list)
        for idx, t in enumerate(time_list):
            day_key = t.date()
            day_index_map[day_key].append(idx)
    
        row_list = []
        for day, idx_list in day_index_map.items():
            one_day_row = {
                "all_day": pd.to_datetime(day),
                "latitude": data["latitude"],
                "longitude": data["longitude"],
                "elevation": data["elevation"],
                "generationtime_ms": data["generationtime_ms"],
            }
            # 截取这一天对应小时数组
            for col in WEATHER_API_CONFIG["hourly_vars"]:
                arr = hourly[col]
                one_day_row[col] = [arr[i] for i in idx_list]
            row_list.append(one_day_row)
    
        # 合并多天，每一天一行
        df_final = pd.DataFrame(row_list)
        # print(f"总天数：{len(df_final)} {df_final.shape}")
        return df_final
    #存天气数据（外层带断线重连重试; 实际写库在 _save_data_impl）
    def save_data(self,df=None,start_date="2026-01-01",end_date=None,on_conflict='update'):
        for _ in range(2):
            try:
                self._ensure_conn()
                return self._save_data_impl(df=df, start_date=start_date,
                                            end_date=end_date, on_conflict=on_conflict)
            except psycopg2.OperationalError as e:
                print(f"  ⚠ 天气库连接异常: {str(e).splitlines()[0]} → 重连重试")
                try:
                    self.conn.close()
                except Exception:
                    pass
                self.conn = None
        print("  ⚠ 天气库重连后仍失败, 本次落地放弃(不阻塞控制)")
        return 0

    def _save_data_impl(self,df=None,start_date="2026-01-01",end_date=None,on_conflict='update'):
        if df is None:
            df=self.get_weather_data(start_date=start_date,end_date=end_date)   
        if df is None or len(df)==0:
            print('weather data没有数据需要保存！！')
            return 0
        df = df.drop_duplicates(subset=["all_day"], keep="last")

        cols = self.weather_table_cols
        col_str = ", ".join(f'"{c}"' for c in cols)
        if on_conflict == "update":
            upd = [c for c in cols if c != "all_day"]
            conflict = ('ON CONFLICT ("all_day") DO UPDATE SET ' +
                        ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in upd))
        else:
            conflict = 'ON CONFLICT ("all_day") DO NOTHING'
        sql = f"INSERT INTO {self.WEATHER_TABLE} ({col_str}) VALUES %s {conflict}"

        records = []
        for _, row in df.iterrows():
            vals = []
            for c in cols:
                v = row.get(c)
                if c in self.INT_ARRAY_COLS and isinstance(v, list):
                    v = [int(x) if x is not None else None for x in v]

                vals.append(DB_RollingResult._to_python_type(v))
            records.append(tuple(vals))

        cur = self.conn.cursor()
        execute_values(cur, sql, records, page_size=100)
        cur.close()
        print(f"weather_data: 写入/更新 {len(records)} 天")
        return len(records)




if __name__=="__main__":
    # db = Real_time_db()
    # print(db.get_current_max_temp(as_of=datetime.datetime.now())) 
    # t1=datetime.datetime.now()
    # print(t1)
    # print(db.get_real_power(as_of=t1))
    wd=Weather_data()
    wd.save_data(end_date=datetime.datetime.now().strftime("%Y-%m-%d"))