import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import re
import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import datetime
import random
import time
import os
import subprocess
import base64
import json
import hashlib
import requests
import psycopg2
from psycopg2.extras import execute_values
import logging


URL = "https://pm.gd.csg.cn/portal/#/home/marketOperateOrg/publicInfo/transaction"
DEBUG_PORT = 9222

# 截图保存根目录
SCREENSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")

# 城市位置配置文件
CITY_POSITIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "city_positions.json")

# 广东21个城市及其大致地理位置（用于辅助 LLM 识别）
GUANGDONG_CITIES = {
    "广州": "中部，省会",
    "深圳": "南部沿海，毗邻香港",
    "珠海": "南部沿海，毗邻澳门",
    "汕头": "东部沿海",
    "佛山": "中部，广州西侧",
    "韶关": "北部内陆",
    "湛江": "西南部沿海",
    "肇庆": "西部内陆",
    "江门": "中南部，珠江口西岸",
    "茂名": "西南部沿海",
    "阳江": "中南部沿海",
    "惠州": "中东部，珠江口东岸",
    "梅州": "东北部内陆",
    "汕尾": "东南部沿海",
    "河源": "东北部内陆",
    "清远": "北部，广州北侧",
    "东莞": "中南部，广州与深圳之间",
    "中山": "中南部，珠江口西岸",
    "潮州": "东部内陆",
    "揭阳": "东部沿海",
    "云浮": "西部内陆"
}

# 城市名规范化：支持中文名（带/不带"市"）+ 英文名，统一映射为英文
CITY_ZH_TO_EN = {
    # 中文名（不带"市"）
    "广州": "Guangzhou", "深圳": "Shenzhen", "珠海": "Zhuhai",
    "汕头": "Shantou",  "佛山": "Foshan",   "韶关": "Shaoguan",
    "湛江": "Zhanjiang","肇庆": "Zhaoqing", "江门": "Jiangmen",
    "茂名": "Maoming",  "阳江": "Yangjiang","惠州": "Huizhou",
    "梅州": "Meizhou",  "汕尾": "Shanwei",  "河源": "Heyuan",
    "清远": "Qingyuan", "东莞": "Dongguan", "中山": "Zhongshan",
    "潮州": "Chaozhou", "揭阳": "Jieyang",  "云浮": "Yunfu",
}
# 自动生成带"市"的映射
for _zh, _en in list(CITY_ZH_TO_EN.items()):
    CITY_ZH_TO_EN[_zh + "市"] = _en
# 英文名自己映射自己（大小写不敏感容错）
for _en in list(CITY_ZH_TO_EN.values()):
    CITY_ZH_TO_EN[_en] = _en
    CITY_ZH_TO_EN[_en.lower()] = _en


def normalize_city_name(name):
    """
    把各种写法的城市名统一为英文（不带"市"）。
    支持：中文、中文+市、英文、大小写不同的英文。
    未知名称原样返回（避免丢数据）。
    """
    if not name:
        return name
    stripped = name.strip()
    # 直接匹配
    if stripped in CITY_ZH_TO_EN:
        return CITY_ZH_TO_EN[stripped]
    # 容错：小写匹配
    if stripped.lower() in CITY_ZH_TO_EN:
        return CITY_ZH_TO_EN[stripped.lower()]
    # 没匹配上，返回原名并告警
    print(f"⚠️  未知城市名: [{name}]，保留原样")
    return stripped


def normalize_price_type(pt):
    """
    把电价类型统一为 'peak' 或 'valley'。
    兼容：peak, valley, trough, 高峰, 低谷。
    """
    if not pt:
        return pt
    mapping = {
        "peak": "peak",
        "valley": "valley",
        "trough": "valley", 
        "高峰": "peak",
        "低谷": "valley",
    }
    stripped = pt.strip()
    return mapping.get(stripped, mapping.get(stripped.lower(), stripped))

# Qwen Vision API 配置
QWEN_API_CONFIG = {
    "base_url": "",
    "api_key": "",
    "model": "qwen-vl-max"  # 视觉模型
}

# TimescaleDB 数据库配置
DB_CONFIG = {
    "timezone":    "Asia/Shanghai",
    "ts_host":     "47.107.118.226",
    "ts_port":     5432,
    "ts_user":     "FrankS",
    "ts_password": "Yuanji!!",
    "ts_dbname":   "energy",
    "measurement": "electricity_price_v3",   # 清洗后的新表（英文城市名 + peak/valley）
}

# 数据库日志
db_log = logging.getLogger("db_writer")


# =====================================================================
#  数据新鲜度校验 & 反爬恢复 相关配置
# =====================================================================
# 是否启用"陈旧数据"检测（高峰==低谷、或与上一天完全相同 → 判为页面未刷新）
ENABLE_STALE_CHECK = True

# 连续处理多少个日期后，主动 reload 一次页面（刷新反爬 token）
# ⚠️ 默认 0（关闭）：本站瑞数 WAF 很凶，reload 会重新触发反爬挑战，
#    很可能被拦成 400 直接把页面搞死（这正是之前第 481 天后全部超时的元凶）。
#    页面死掉由 ensure_page_alive 兜底恢复，不需要主动定期 reload。
#    仅当你确认 reload 不会被拦时，才把它设成 20 之类的值。
RELOAD_EVERY_N_DATES = 0

# 单个日期采集失败（数据陈旧 / WAF 拦截）时的最大重试次数（每次重试前会 reload 页面）
MAX_ATTEMPTS_PER_DATE = 2

# 识别到真正的电价数据接口后，可把它的 URL 关键字填在这里（如 "nodePrice"）。
# 设置后，切换日期会显式等待该接口返回 200，比固定 sleep 更可靠。留空则不启用。
DATA_API_KEYWORD = None

# 采集方式：True = 优先从 ECharts 实例直读（推荐，稳）；
#           False = 强制走旧的 tooltip 悬停方案
PREFER_ECHARTS = True

# 页面健康状态（在 attach_diagnostics 的网络回调里实时更新）
PAGE_STATE = {
    "waf_block_count": 0,   # 采集期间遇到的 WAF 拦截（HTTP 4xx）次数
    "reload_needed": False, # 是否需要 reload 恢复
    "data_api_status": None,# 最近一次数据接口的 HTTP 状态码
    "seen_urls": [],        # 采集期间出现过的所有 XHR/fetch URL（--probe 用来定位数据接口）
}


def reset_page_state():
    """每个日期开始采集前重置页面健康状态（seen_urls 不在此清空，供 --probe 累积）"""
    PAGE_STATE["waf_block_count"] = 0
    PAGE_STATE["reload_needed"] = False
    PAGE_STATE["data_api_status"] = None


def random_sleep(min_sec: float = 0.6, max_sec: float = 1.8):
    time.sleep(random.uniform(min_sec, max_sec))


def create_date_dir(date_str):
    """
    创建日期文件夹并返回路径
    date_str: "20260701" 格式
    返回: "C:\...\screenshots\20260701\"
    """
    dir_path = os.path.join(SCREENSHOT_DIR, date_str)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def get_screenshot_path(date_str, filename):
    """
    获取截图保存路径
    date_str: "20260701"
    filename: "map_peak.png"
    返回: "C:\...\screenshots\20260701\map_peak.png"
    """
    dir_path = create_date_dir(date_str)
    return os.path.join(dir_path, filename)


# =====================================================================
#  Phase 1: Tab 切换 + 地图加载
# =====================================================================

def cleanup_echarts(page):
    """清理残留的 ECharts 实例，防止重复初始化导致 tooltip 异常"""
    try:
        page.evaluate("""() => {
            // 方法1：通过全局 echarts 对象销毁
            if (typeof echarts !== 'undefined') {
                const doms = document.querySelectorAll('[data-echarts-instance], [_echarts_instance_]');
                doms.forEach(dom => {
                    try { echarts.dispose(dom); } catch(e) {}
                });
            }
            // 方法2：清除 #geoChart 内残留的 canvas（防止叠加）
            const chart = document.getElementById('geoChart');
            if (chart) {
                chart.querySelectorAll('canvas').forEach(c => c.remove());
            }
        }""")
        random_sleep(0.3, 0.5)
    except Exception as e:
        print(f"⚠️  清理 ECharts 实例失败（非致命）：{e}")

def click_peak_tab(page):
    """点击高峰时刻标签（模拟真人）"""
    print("📊 切换到【高峰时刻日前节点电价】...")
    cleanup_echarts(page)
    btn = page.locator('button:has-text("高峰时刻日前节点电价")')
    human_click(page, btn, delay_before=(0.4, 0.8), delay_after=(0.8, 1.5))
    wait_for_map_load(page)

def click_valley_tab(page):
    """点击低谷时刻标签（模拟真人）"""
    print("📊 切换到【低谷时刻日前节点电价】...")
    cleanup_echarts(page)
    btn = page.locator('button:has-text("低谷时刻日前节点电价")')
    human_click(page, btn, delay_before=(0.4, 0.8), delay_after=(0.8, 1.5))
    wait_for_map_load(page)

def wait_for_map_load(page, timeout=10):
    """等待地图加载完成"""
    try:
        page.locator('#geoChart').wait_for(state='visible', timeout=timeout * 1000)
        random_sleep(2.0, 3.0)  # 等待 ECharts 完全渲染
        print("✅ 地图加载完成")
    except Exception as e:
        print(f"⚠️  地图加载超时：{e}")


# =====================================================================
#  Phase 2: Qwen API + 地图截图识别
# =====================================================================

def screenshot_map(page, save_path):
    """截图地图元素"""
    map_elem = page.locator('#geoChart')
    map_elem.screenshot(path=save_path)
    print(f"📸 地图已截图：{save_path}")
    return save_path

def call_qwen_vision(image_path, prompt):
    """
    调用 Qwen 视觉模型分析图片
    返回: 模型的文本回复
    """
    # 1. 读取图片并转 base64
    with open(image_path, 'rb') as f:
        image_base64 = base64.b64encode(f.read()).decode('utf-8')

    # 2. 构造请求
    url = f"{QWEN_API_CONFIG['base_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {QWEN_API_CONFIG['api_key']}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": QWEN_API_CONFIG['model'],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "max_tokens": 2000
    }

    # 3. 发送请求
    print(f"🤖 调用 Qwen Vision API...")
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()

        # 4. 提取回复文本
        content = result['choices'][0]['message']['content']
        print(f"✅ API 返回成功")
        return content
    except Exception as e:
        print(f"❌ API 调用失败：{e}")
        return None

def identify_cities_on_map(page, date_str, tab_type):
    """
    获取城市位置：优先使用手动标注的配置文件，否则使用 LLM 识别
    返回: [{"name": "广州", "x": 0.65, "y": 0.42}, ...]
    """
    # 1. 检查是否存在手动标注的配置文件
    if os.path.exists(CITY_POSITIONS_FILE):
        print(f"📍 使用手动标注的城市位置配置: {CITY_POSITIONS_FILE}")
        try:
            with open(CITY_POSITIONS_FILE, 'r', encoding='utf-8') as f:
                cities = json.load(f)
            print(f"🗺️  加载 {len(cities)} 个城市位置：{[c['name'] for c in cities]}")
            return cities
        except Exception as e:
            print(f"⚠️  读取配置文件失败: {e}，改用 LLM 识别")

    # 2. 没有配置文件，使用 LLM 识别
    print("🤖 未找到城市位置配置，使用 LLM 识别...")

    # 保存到日期文件夹
    screenshot_path = get_screenshot_path(date_str, f"map_{tab_type}.png")
    screenshot_map(page, screenshot_path)

    # 优化的 prompt：提供广东21个城市的地理知识
    city_list = "\n".join([f"- {name}：{location}" for name, location in GUANGDONG_CITIES.items()])

    prompt = f"""这是一个广东省电力节点电价地图。地图上标注了广东省的21个地级市。

广东省21个城市及其地理位置如下：
{city_list}

请根据以上地理分布知识，识别地图上的所有城市标记点，并给出每个城市在图片中的相对坐标。

要求：
1. 坐标格式：x 和 y 是 0-1 之间的小数，左上角为 (0,0)，右下角为 (1,1)
2. 必须识别全部21个城市
3. 根据地理位置关系验证你的识别（如：深圳应该在广州的东南方，东莞在广州和深圳之间）
4. 返回严格的 JSON 格式

返回 JSON 格式：
[
  {{"name": "广州", "x": 0.xx, "y": 0.xx}},
  {{"name": "深圳", "x": 0.xx, "y": 0.xx}},
  ...
]

只返回 JSON 数组，不要其他内容。"""

    result_text = call_qwen_vision(screenshot_path, prompt)
    if not result_text:
        return []

    # 解析 JSON（处理可能的 markdown 代码块）
    try:
        json_str = result_text.strip()
        if json_str.startswith('```'):
            json_str = re.sub(r'```json\n?', '', json_str)
            json_str = json_str.replace('```', '')

        cities = json.loads(json_str)
        print(f"🗺️  LLM 识别到 {len(cities)} 个城市：{[c['name'] for c in cities]}")

        # 验证是否识别了所有21个城市
        identified_names = [c['name'] for c in cities]
        missing = [name for name in GUANGDONG_CITIES.keys() if name not in identified_names]
        if missing:
            print(f"⚠️  缺少城市：{missing}")
            print(f"💡 建议使用 annotate_cities.py 工具手动标注城市位置")

        return cities
    except Exception as e:
        print(f"❌ 解析城市列表失败：{e}")
        print(f"原始返回：{result_text}")
        return []


# =====================================================================
#  Phase 3: 鼠标悬停 + Tooltip 截图
# =====================================================================

def hover_and_screenshot_tooltip(page, city_info, date_str, tab_type, retry_count=0):
    """
    模拟真人悬停到指定坐标，截图 tooltip
    city_info: {"name": "广州", "x": 0.65, "y": 0.42}
    retry_count: 当前重试次数（仅用于日志）
    返回: tooltip 截图路径，失败返回 None
    """
    city_name = city_info['name']
    if retry_count > 0:
        print(f"\n🔄 [{retry_count}/3] 重试：{city_name}")
    else:
        print(f"\n🎯 处理城市：{city_name} (相对坐标: {city_info['x']:.2f}, {city_info['y']:.2f})")

    # 1. 获取地图元素的边界框
    map_elem = page.locator('#geoChart')
    box = map_elem.bounding_box()
    if not box:
        print(f"❌ 无法获取地图边界框")
        return None

    # 2. 计算实际像素坐标（固定偏移范围）
    target_x = box['x'] + box['width'] * city_info['x'] + random.uniform(-2, 2)
    target_y = box['y'] + box['height'] * city_info['y'] + random.uniform(-2, 2)

    # 3. 模拟真人鼠标移动
    current_x = box['x'] - 50
    current_y = box['y'] - 50

    steps = random.randint(8, 15)
    for i in range(1, steps + 1):
        progress = i / steps
        jitter_x = random.uniform(-5, 5) * (1 - progress)
        jitter_y = random.uniform(-3, 3) * (1 - progress)

        step_x = current_x + (target_x - current_x) * progress + jitter_x
        step_y = current_y + (target_y - current_y) * progress + jitter_y

        page.mouse.move(step_x, step_y)
        random_sleep(0.02, 0.06)

    page.mouse.move(target_x, target_y)
    random_sleep(0.8, 1.2)

    # 4. 等待 tooltip 出现
    tooltip_selectors = [
        '.ant-tooltip',
        '.echarts-tooltip',
        '[class*="tooltip"]',
        'div[style*="position: absolute"]'
    ]

    tooltip = None
    for selector in tooltip_selectors:
        try:
            tooltip = page.locator(selector).first
            tooltip.wait_for(state='visible', timeout=2000)
            break
        except:
            tooltip = None
            continue

    if not tooltip:
        print(f"⚠️  {city_name} 的 tooltip 未出现")
        return None

    # 5. 获取 tooltip 数据（三级降级）
    tooltip_filename = f"tooltip_{tab_type}_{city_name}.png"
    tooltip_path = get_screenshot_path(date_str, tooltip_filename)

    # 方案 A：直接从 DOM 提取文本（最快，无需截图和 OCR）
    try:
        text = tooltip.evaluate("el => el.innerText || el.textContent || ''").strip()
        if text and len(text) > 3:
            print(f"📝 直接从 DOM 提取到文本：{text[:80]}")
            data = extract_price_from_text(text, city_name)
            if data:
                # 顺便保存截图用于事后核查（best-effort，不影响主流程）
                try:
                    tooltip.screenshot(path=tooltip_path, timeout=3000)
                    print(f"📸 Tooltip 截图已保存：{tooltip_path}")
                except:
                    pass
                return data
            # 解析失败，继续走截图方案
            print(f"⚠️  DOM 文本解析失败，降级到截图方案")
    except:
        pass

    # 方案 B：元素截图（超时缩短到 3 秒快速失败）
    try:
        tooltip.screenshot(path=tooltip_path, timeout=3000)
        print(f"📸 Tooltip 已截图：{tooltip_path}")
        return tooltip_path
    except Exception as e:
        print(f"⚠️  元素截图失败，尝试降级方案：{e}")

    # 方案 C：用 JS 获取 tooltip 坐标 + 整页截图裁剪（绕过 stability 检查）
    try:
        box = tooltip.evaluate("""el => {
            const rect = el.getBoundingClientRect();
            return {x: rect.x, y: rect.y, width: rect.width, height: rect.height};
        }""")
        if box and box['width'] > 0 and box['height'] > 0:
            padding = 5
            page.screenshot(path=tooltip_path, clip={
                'x': max(0, box['x'] - padding),
                'y': max(0, box['y'] - padding),
                'width': box['width'] + padding * 2,
                'height': box['height'] + padding * 2,
            })
            print(f"📸 Tooltip 已截图（降级方案）：{tooltip_path}")
            return tooltip_path
        else:
            print(f"❌ 降级方案失败：tooltip 尺寸无效 ({box})")
            return None
    except Exception as e2:
        print(f"❌ 降级截图也失败：{e2}")
        return None

def extract_price_from_tooltip(tooltip_path, city_name):
    """用 LLM 从 tooltip 截图中提取电价"""
    prompt = f"""这是 {city_name} 的电价信息截图。

请仔细识别并提取以下信息：
1. 城市名称：完整读取截图中的城市名称
2. 电价数值：精确读取数字，保留所有小数位，不要四舍五入或简化
3. 单位：完整读取截图中的单位,如 "元/MWh"

重要提醒：
- 请逐字逐句识别截图中的内容，不要自行推断或修改
- 如果截图显示 "313.041 元/MWh"，必须返回 313.041 和 元/MWh
- 如果截图显示 "0.500 元/MWh"，必须返回 0.500 和 元/MWh
- 严格按照截图中的实际内容提取，不要添加或遗漏任何信息

返回 JSON 格式：{{"city": "城市名", "price": 数值, "unit": "单位"}}
只返回 JSON，不要其他内容。"""

    result_text = call_qwen_vision(tooltip_path, prompt)
    if not result_text:
        return None

    try:
        # 解析 JSON
        json_str = result_text.strip()
        if json_str.startswith('```'):
            json_str = re.sub(r'```json\n?', '', json_str)
            json_str = json_str.replace('```', '')

        data = json.loads(json_str)
        # 城市名规范化为英文（LLM 可能返回中文或英文）
        if 'city' in data:
            data['city'] = normalize_city_name(data['city'])
        print(f"💰 提取成功：{data.get('city', city_name)} = {data.get('price')} {data.get('unit')}")
        return data
    except Exception as e:
        print(f"❌ 解析电价失败：{e}")
        print(f"原始返回：{result_text}")
        return None


def extract_price_from_text(text, city_name):
    """
    从 tooltip DOM 文本中解析电价（无需 OCR）
    典型格式: "广州\n313.041\n元/MWh" 或 "节点电价\n313.041 元/MWh"
    新格式:   "中山市\n808.7488 (元/MWh)"  ← 单位带括号
    """
    # 提取数字（电价通常是较大的浮点数，如 313.041）
    numbers = re.findall(r'(\d+\.?\d*)', text)
    # 过滤掉明显不是电价的数字（如 0、1、2 这种序号）
    prices = [float(n) for n in numbers if float(n) > 1]

    # 提取单位（去除首尾可能包裹的括号，如 "(元/MWh)" -> "元/MWh"）
    unit_match = re.search(r'[()]*(元/[^\\n\s,，()]+)[)]*', text)
    unit = unit_match.group(1) if unit_match else "元/MWh"
    # 再次保险：去掉残留的括号
    unit = unit.strip("()")

    # 城市名统一为英文（不带"市"）
    city = normalize_city_name(city_name)

    if prices:
        price = prices[-1]  # 取最后一个（通常是电价值）
        print(f"💰 DOM 文本提取成功：{city} = {price} {unit}")
        return {"city": city, "price": price, "unit": unit}
    else:
        print(f"⚠️  DOM 文本中未找到有效电价数字：{text[:60]}")
        return None


# =====================================================================
#  Phase 4: 完整流程编排
# =====================================================================

def process_map_data(page, date_str, date_formatted, tab_type):
    """
    处理单个标签的地图数据（带重试机制和实时保存）
    date_str: "20260702" 格式（用于文件命名）
    date_formatted: "2026-07-02" 格式（用于日志和数据库记录）
    tab_type: "peak" 或 "valley"
    """
    MAX_RETRIES = 3  # 最大重试次数
    price_type = "高峰" if tab_type == "peak" else "低谷"

    # 1. 识别城市
    cities = identify_cities_on_map(page, date_str, tab_type)
    if not cities:
        print("⚠️  未识别到任何城市")
        return []

    # 2. 逐个悬停并截图（带重试）
    results = []
    failed_cities = []

    for city in cities:
        city_name = city['name']
        success = False

        # 重试循环
        for retry in range(MAX_RETRIES):
            # 尝试悬停和截图
            result_or_path = hover_and_screenshot_tooltip(page, city, date_str, tab_type, retry_count=retry)

            if not result_or_path:
                # Tooltip 未出现或截图失败
                if retry < MAX_RETRIES - 1:
                    print(f"⏳ {city_name} 准备重试...")
                    random_sleep(1.5, 2.5)  # 重试前等待
                    continue
                else:
                    print(f"❌ {city_name} 截图失败（已重试 {MAX_RETRIES} 次）")
                    failed_cities.append(city_name)
                    break

            # 如果是直接从 DOM 提取的数据，跳过 OCR
            if isinstance(result_or_path, dict):
                results.append(result_or_path)
                success = True
                break

            # 否则是截图路径，用 OCR 提取
            data = extract_price_from_tooltip(result_or_path, city_name)

            if data:
                results.append(data)
                success = True
                break
            else:
                # 数据提取失败
                if retry < MAX_RETRIES - 1:
                    print(f"⏳ {city_name} 数据提取失败，准备重试...")
                    random_sleep(1.5, 2.5)
                    continue
                else:
                    print(f"❌ {city_name} 数据提取失败（已重试 {MAX_RETRIES} 次）")
                    failed_cities.append(city_name)
                    break

        # 每个城市处理完后等待
        if success:
            random_sleep(1.0, 1.5)
        else:
            random_sleep(0.5, 1.0)  # 失败后等待稍短

    # 3. 输出统计信息
    if failed_cities:
        print(f"\n⚠️  失败城市（{len(failed_cities)}/{len(cities)}）：{', '.join(failed_cities)}")
    else:
        print(f"\n✅ 成功采集所有 {len(cities)} 个城市数据")

    return results


# =====================================================================
#  Phase 3.5: ECharts 数据直读（替代 tooltip 悬停，一次拿到全部城市）
# =====================================================================

def extract_prices_via_echarts(page, tab_type):
    """
    直接从 ECharts 实例读取地图上的全部城市电价，避免逐个悬停 tooltip
    （tooltip 方案会读到冻结的旧 div，是"数据永远不变"的元凶之一）。

    返回: [{"city": "Shenzhen", "price": 547.9389, "unit": "元/MWh"}, ...]
          读取失败返回 None（由调用方降级到 tooltip 悬停方案）
    """
    js = r"""
    () => {
        try {
            if (typeof echarts === 'undefined') return {error: 'echarts 未定义'};
            const dom = document.getElementById('geoChart');
            if (!dom) return {error: '未找到 #geoChart 元素'};
            const chart = echarts.getInstanceByDom(dom);
            if (!chart) return {error: '#geoChart 上没有 ECharts 实例'};
            const opt = chart.getOption();
            const seriesList = opt.series || [];
            const picked = [];
            for (const s of seriesList) {
                const data = s.data || s.dataList;
                if (!Array.isArray(data)) continue;
                for (const d of data) {
                    if (!d) continue;
                    let name = d.name;
                    let value = d.value;
                    // value 可能是 [lng, lat, price] 这样的数组，取最后一个数值
                    if (Array.isArray(value)) value = value[value.length - 1];
                    if (name == null || value == null) continue;
                    const num = Number(value);
                    if (!isFinite(num)) continue;
                    picked.push({name: String(name), value: num});
                }
            }
            return {data: picked, seriesCount: seriesList.length};
        } catch (e) {
            return {error: String(e)};
        }
    }
    """
    try:
        raw = page.evaluate(js)
    except Exception as e:
        print(f"⚠️  ECharts 直读异常：{e}")
        return None

    if not raw or raw.get("error"):
        print(f"⚠️  ECharts 直读失败：{raw.get('error') if raw else '无返回'}")
        return None

    data = raw.get("data") or []
    if not data:
        print(f"⚠️  ECharts 直读到 0 条数据（series 数={raw.get('seriesCount')}）")
        return None

    results = []
    seen = set()
    for item in data:
        city = normalize_city_name(item["name"])
        if city in seen:
            continue
        seen.add(city)
        results.append({"city": city, "price": item["value"], "unit": "元/MWh"})

    sample = results[0]
    print(f"💠 ECharts 直读成功（{tab_type}）：{len(results)} 个城市，示例 {sample['city']} = {sample['price']}")
    return results


def collect_tab(page, date_str, date_formatted, tab_type):
    """采集单个 tab：优先 ECharts 直读，失败（或被禁用）再降级到 tooltip 悬停方案"""
    if PREFER_ECHARTS:
        data = extract_prices_via_echarts(page, tab_type)
        if data:
            return data
        print(f"↩️  ECharts 直读不可用，降级到 tooltip 悬停方案（{tab_type}）...")
    return process_map_data(page, date_str, date_formatted, tab_type)


# =====================================================================
#  Phase 3.6: 数据新鲜度校验（防止把冻结的旧数据写进数据库）
# =====================================================================

def compute_day_signature(peak_data, valley_data):
    """把一天的数据压成一个可比较的签名，用于检测"跨天数据是否完全相同" """
    def sig(data):
        return sorted(
            (d.get("city"), round(float(d.get("price")), 4))
            for d in (data or [])
            if d.get("price") is not None
        )
    return (sig(peak_data), sig(valley_data))


def validate_day_data(date_formatted, peak_data, valley_data, prev_signature):
    """
    校验当天采集到的数据是否可信。
    返回: (is_valid: bool, reason: str, signature)
    """
    if not peak_data or not valley_data:
        return False, "高峰或低谷数据为空", None

    signature = compute_day_signature(peak_data, valley_data)

    if ENABLE_STALE_CHECK:
        # 1) 高峰与低谷完全相同 —— 物理上不可能，说明 tab 切换没生效
        if signature[0] and signature[0] == signature[1]:
            return False, "高峰数据与低谷数据完全相同（tab 切换未生效/页面数据冻结）", signature

        # 2) 与上一个成功采集的日期完全相同 —— 说明日期切换没生效
        if prev_signature is not None and signature == prev_signature:
            return False, "与上一采集日数据完全相同（页面数据未随日期刷新）", signature

    return True, "", signature


# =====================================================================
#  Phase 3.7: 反爬恢复（reload 刷新 token）
# =====================================================================

def reload_and_wait(page):
    """重新加载页面，让反爬（瑞数）JS 重新生成 token，恢复被 WAF 拦截的数据接口。
    注意：reload 会重新触发反爬挑战，可能失败——所以用"日期选择器可见"严格验收。"""
    print("🔄 重新加载页面（尝试刷新反爬 token）...")
    try:
        page.reload(wait_until="domcontentloaded", timeout=30000)
    except Exception as e:
        print(f"⚠️  reload 异常：{e}")
    if page_is_ready(page, timeout_ms=20000):
        reset_page_state()
        random_sleep(2, 3)
        print("✅ 页面已重新加载")
        return True
    print("⚠️  reload 后日期选择器未出现，页面可能被反爬拦截")
    return False


def dump_echarts_structure(page):
    """[--probe 用] 打印 #geoChart 上 ECharts 实例的 series 结构，帮助确认直读是否取到正确数据"""
    js = r"""
    () => {
        try {
            if (typeof echarts === 'undefined') return {error: 'echarts 未定义'};
            const dom = document.getElementById('geoChart');
            if (!dom) return {error: '未找到 #geoChart'};
            const chart = echarts.getInstanceByDom(dom);
            if (!chart) return {error: '无 ECharts 实例'};
            const opt = chart.getOption();
            const summary = (opt.series || []).map((s, idx) => {
                const data = s.data || s.dataList || [];
                const sample = Array.isArray(data) ? data.slice(0, 3) : [];
                return {
                    idx, type: s.type, name: s.name || '(无name)',
                    dataLen: Array.isArray(data) ? data.length : 'N/A',
                    sample: sample,
                };
            });
            return {seriesCount: (opt.series || []).length, series: summary};
        } catch (e) { return {error: String(e)}; }
    }
    """
    try:
        info = page.evaluate(js)
    except Exception as e:
        print(f"❌ dump 失败：{e}")
        return
    print("\n" + "-" * 50)
    print("🔬 ECharts series 结构：")
    print(json.dumps(info, ensure_ascii=False, indent=2, default=str))
    print("-" * 50 + "\n")


def run_probe(page, year, month, day):
    """
    [--probe] 探测模式：只处理一个日期，抓取所有 XHR 响应体，
    自动找出"返回城市电价数据"的接口（用于填 DATA_API_KEYWORD），
    并确认高峰/低谷数据是否随操作变化。不写数据库。
    """
    print("\n" + "=" * 60)
    print(f"🔬 PROBE 探测模式：{year}-{month:02d}-{day:02d}（不写数据库）")
    print("=" * 60)

    # 本地捕获 Response 对象（先只存对象，流程结束后再读 body，避免在事件回调里阻塞）
    captured = []

    def _capture(response):
        try:
            if response.request.resource_type in ("xhr", "fetch"):
                captured.append(response)
        except Exception:
            pass

    page.on("response", _capture)
    reset_page_state()

    try:
        select_calendar_date(page, year, month, day)
        random_sleep(3, 4)

        print("\n>>> 切到【高峰】...")
        click_peak_tab(page)
        dump_echarts_structure(page)
        random_sleep(2, 3)

        print("\n>>> 切到【低谷】...")
        click_valley_tab(page)
        random_sleep(2, 3)
    finally:
        try:
            page.remove_listener("response", _capture)
        except Exception:
            pass

    # 读取所有捕获到的响应体，按"包含多少广东城市名"打分
    print("\n" + "=" * 60)
    print(f"🌐 共捕获 {len(captured)} 个 XHR/fetch 响应，正在分析哪个是电价数据接口...")
    print("=" * 60)

    city_names = list(GUANGDONG_CITIES.keys())
    scored = []  # (city_hits, status, url, body_text)

    for r in captured:
        url = r.url
        try:
            status = r.status
        except Exception:
            status = "?"
        body = ""
        try:
            body = r.text()
        except Exception as e:
            body = f"<body 读取失败: {e}>"
        hits = sum(1 for c in city_names if c in body)
        scored.append((hits, status, url, body))

    # 城市命中数高的排前面（最可能是电价数据接口）
    scored.sort(key=lambda x: x[0], reverse=True)

    shown = set()
    for hits, status, url, body in scored:
        if url in shown:
            continue
        shown.add(url)
        flag = "  ⭐ 极可能是电价接口" if hits >= 5 else ("  ← 含城市名" if hits > 0 else "")
        print(f"\n[HTTP {status}] 城市命中={hits}{flag}")
        print(f"  URL: {url}")
        if hits > 0 or len(body) < 300:
            preview = body[:800].replace("\n", " ")
            print(f"  BODY: {preview}")

    print("\n" + "=" * 60)
    print("💡 下一步：")
    print("  1) 找到上面标 ⭐ 的接口，把它 URL 里的独特片段（如某个路径词）填入脚本顶部 DATA_API_KEYWORD")
    print("  2) 把该接口的完整 BODY 结构发给我，我据此写精确的 JSON 解析，实现网络直采（绕开 echarts/tooltip）")
    print("=" * 60 + "\n")


def process_one_date(page, year, month, day, db_conn=None, db_cfg=None, prev_signature=None):
    """
    处理单个日期的完整流程（含 WAF 恢复 + 数据新鲜度校验）

    db_conn:        数据库连接（可选，传入则实时写入数据库）
    db_cfg:         数据库配置（可选）
    prev_signature: 上一个成功采集日期的数据签名（用于检测"跨天数据冻结"）

    返回 dict 中新增字段：
      valid:     True = 数据通过校验并已尝试写库；False = 数据不可信，已跳过写库
      signature: 本天数据签名（供下一天做新鲜度对比）
      skipped:   True = 数据库已存在该日期，直接跳过
    """
    date_str = f"{year}{month:02d}{day:02d}"
    date_formatted = f"{year}-{month:02d}-{day:02d}"

    # 0. 去重检查（采集前查数据库，避免无谓采集）
    if DEDUPE_MODE == "skip":
        if db_conn is not None and db_cfg is not None:
            if check_date_exists_in_db(db_conn, db_cfg, date_formatted):
                print(f"⏭️  日期 {date_formatted} 已在数据库中存在，跳过采集（当前模式：不覆盖）")
                return {
                    "date": date_formatted, "peak": [], "valley": [],
                    "skipped": True, "valid": False,
                    "signature": prev_signature, "_db_conn": db_conn,
                }
        else:
            print("⚠️  数据库连接不可用，跳过去重检查")

    peak_data, valley_data, signature = [], [], None

    for attempt in range(1, MAX_ATTEMPTS_PER_DATE + 1):
        # 每次尝试前重置页面健康状态
        reset_page_state()

        # 1. 选择日期（模拟真人点击）
        #    若配置了数据接口关键字，用 expect_response 包裹点击动作，
        #    显式等待该接口返回（比固定 sleep 更可靠）
        if DATA_API_KEYWORD:
            try:
                with page.expect_response(
                    lambda r: DATA_API_KEYWORD in r.url, timeout=15000
                ) as resp_info:
                    select_calendar_date(page, year, month, day)
                resp = resp_info.value
                PAGE_STATE["data_api_status"] = resp.status
                print(f"📡 数据接口返回 HTTP {resp.status}")
                random_sleep(1, 2)
            except Exception as e:
                print(f"⚠️  等待数据接口超时/失败（日期已点击，继续）：{e}")
                random_sleep(2, 4)
        else:
            select_calendar_date(page, year, month, day)
            random_sleep(2, 4)  # 等待页面数据加载

        # 2. 处理高峰时刻
        click_peak_tab(page)
        peak_data = collect_tab(page, date_str, date_formatted, "peak")

        # 3. 处理低谷时刻
        click_valley_tab(page)
        valley_data = collect_tab(page, date_str, date_formatted, "valley")

        # 4. WAF 拦截信息（仅记录，不据此 reload —— reload 在本站极易被拦并把页面搞死）
        if PAGE_STATE["reload_needed"]:
            print(f"⚠️  采集期间数据接口返回 {PAGE_STATE['waf_block_count']} 次 4xx，数据可能不可信")

        # 5. 数据新鲜度校验
        is_valid, reason, signature = validate_day_data(
            date_formatted, peak_data, valley_data, prev_signature
        )
        if is_valid:
            print(f"✅ 数据校验通过：{date_formatted}")
            break

        print(f"❌ 数据校验失败：{reason}")
        if attempt < MAX_ATTEMPTS_PER_DATE:
            # 不 reload（reload 会重新触发反爬挑战、大概率被 WAF 拦成 400 把页面搞死）。
            # 只重新选日期 + 重读数据，应对偶发性的数据未刷新。
            print("🔄 重新选择日期后重试（不 reload，避免触发反爬）...")
            random_sleep(2, 3)
            continue

        # 重试用尽仍不可信 —— 不写库，避免污染数据库
        print(f"⚠️  已重试 {MAX_ATTEMPTS_PER_DATE} 次，{date_formatted} 数据仍不可信，跳过写库")
        return {
            "date": date_formatted, "peak": peak_data, "valley": valley_data,
            "valid": False, "reason": reason,
            "signature": prev_signature, "_db_conn": db_conn,
        }

    # 6. 实时写入数据库（仅当数据通过校验时才会走到这里）
    if db_conn is not None and db_cfg is not None:
        db_conn = ensure_db_connection(db_conn, db_cfg)
        if db_conn is None:
            print(f"❌ 数据库重连失败，日期 {date_formatted} 的采集数据未能保存")
        else:
            write_one_day_to_db(db_conn, db_cfg, date_formatted, peak_data, valley_data, mode=DEDUPE_MODE)
    else:
        print("⚠️  数据库连接不可用，本次采集数据未保存")

    # 7. 每个日期处理完后，像人一样休息一下
    random_sleep(1.5, 3.0)

    return {
        "date": date_formatted,
        "peak": peak_data,
        "valley": valley_data,
        "valid": True,
        "signature": signature,
        "_db_conn": db_conn    # 把（可能已重连的）连接带回去，让主循环拿到新连接
    }

# 去重模式配置
# "skip" = 不覆盖（已有数据则跳过）
# "overwrite" = 覆盖（删除旧数据，保存新数据）
DEDUPE_MODE = "skip"


# =====================================================================
#  数据库写入功能
# =====================================================================

def get_db_connection(cfg: dict):
    """获取数据库连接（开启 TCP keepalive 防止长时间运行被中间网络设备切断）"""
    try:
        conn = psycopg2.connect(
            host=cfg['ts_host'],
            port=cfg['ts_port'],
            dbname=cfg['ts_dbname'],
            user=cfg['ts_user'],
            password=cfg['ts_password'],
            keepalives=1,            # 启用 TCP keepalive
            keepalives_idle=60,      # 空闲 60s 后开始发探测包
            keepalives_interval=10,  # 探测包间隔 10s
            keepalives_count=6,      # 6 次无响应视为断开
        )
        conn.autocommit = True
        print(f"✅ 数据库连接成功：{cfg['ts_host']}:{cfg['ts_port']}/{cfg['ts_dbname']}")
        return conn
    except psycopg2.Error as exc:
        print(f"❌ 数据库连接失败: {exc}")
        return None


def ensure_db_connection(conn, cfg: dict):
    """
    检查数据库连接是否可用，不可用则自动重连。
    返回：可用的连接对象（可能是原连接，也可能是新连接）

    用于长时间运行的采集任务：中间页面抓取 / API 调用耗时较长，
    PostgreSQL 服务端或中间网络设备（SLB、防火墙）可能已切断空闲 TCP 连接，
    但 psycopg2 不一定知道 —— 所以不能只看 conn.closed，还要 SELECT 1 ping 一下。
    """
    need_reconnect = False

    # 1. 明显已关闭
    if conn is None or conn.closed:
        need_reconnect = True
    else:
        # 2. 尝试 ping 数据库（捕获"连接已断但 psycopg2 还不知道"的情况）
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        except Exception as exc:
            print(f"⚠️  数据库连接不可用（{exc}），准备重连...")
            need_reconnect = True
            try:
                conn.close()
            except Exception:
                pass

    if need_reconnect:
        print("🔄 重新建立数据库连接...")
        conn = get_db_connection(cfg)
        if conn:
            ensure_db_table(conn, cfg)

    return conn


def ensure_db_table(conn, cfg: dict):
    """确保数据库表存在（带唯一约束防止重复数据）"""
    table_name = cfg['measurement']
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                time            TIMESTAMPTZ     NOT NULL,
                city            TEXT            NOT NULL,
                price_type      TEXT            NOT NULL,
                price_cny_mwh   DOUBLE PRECISION,
                CONSTRAINT uq_{table_name} UNIQUE (time, city, price_type)
            );
        """)
        # 尝试创建超表（TimescaleDB）
        try:
            cur.execute(f"""
                SELECT create_hypertable('{table_name}', 'time',
                    if_not_exists => TRUE,
                    chunk_time_interval => INTERVAL '30 days'
                );
            """)
        except Exception as exc:
            db_log.warning("超表创建跳过（非 TimescaleDB 或已存在）: %s", exc)

        # 创建索引
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_city_time
                ON {table_name} (city, time DESC);
        """)
    print(f"✅ 数据表 '{table_name}' 已就绪")


def check_date_exists_in_db(conn, cfg: dict, date_str: str) -> bool:
    """
    检查数据库中是否已存在指定日期的数据（用于去重）
    date_str: "2026-07-02" 格式
    返回: True = 已存在，False = 不存在
    """
    table_name = cfg['measurement']
    timezone = cfg['timezone']
    try:
        dt = pd.to_datetime(date_str).tz_localize(timezone)
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT EXISTS (
                    SELECT 1 FROM {table_name}
                    WHERE time = %s::timestamptz
                    LIMIT 1
                )
            """, (dt.to_pydatetime(),))
            exists = cur.fetchone()[0]
        return bool(exists)
    except Exception as exc:
        db_log.warning(f"检查数据库日期存在性失败: {exc}")
        return False


def write_one_day_to_db(conn, cfg: dict, date_str: str,
                         peak_data: list, valley_data: list,
                         mode: str = DEDUPE_MODE) -> bool:
    """
    将单天的数据写入数据库（带去重）

    参数：
        conn: 数据库连接
        cfg: 数据库配置
        date_str: "2026-07-02" 格式的日期
        peak_data: 高峰数据列表 [{"city": "广州市", "price": 456.78, "unit": "元/MWh"}, ...]
        valley_data: 低谷数据列表
        mode: "skip"（已存在则跳过）或 "overwrite"（已存在则覆盖）

    返回：
        True = 写入成功，False = 跳过或失败
    """
    table_name = cfg['measurement']
    timezone = cfg['timezone']

    # 1. 去重检查
    exists = check_date_exists_in_db(conn, cfg, date_str)
    if exists:
        if mode == "skip":
            print(f"⏭️  [DB] 日期 {date_str} 已存在，跳过写入（模式：不覆盖）")
            return False
        elif mode == "overwrite":
            print(f"🔄 [DB] 日期 {date_str} 已存在，准备覆盖（模式：覆盖）")
        else:
            print(f"⚠️  [DB] 未知去重模式: {mode}，按 skip 处理")
            return False

    # 2. 构建写入行（城市名和 price_type 在此统一规范化，作为最后一道防线）
    dt = pd.to_datetime(date_str).tz_localize(timezone)
    ts = dt.to_pydatetime()
    rows = []

    for item in peak_data:
        if item.get('price'):
            city_en = normalize_city_name(item.get('city', ''))
            rows.append((ts, city_en, 'peak', float(item['price'])))

    for item in valley_data:
        if item.get('price'):
            city_en = normalize_city_name(item.get('city', ''))
            rows.append((ts, city_en, 'valley', float(item['price'])))

    if not rows:
        print(f"⚠️  [DB] 日期 {date_str} 没有有效数据可写入")
        return False

    # 3. 写入（覆盖模式下先删除旧数据）
    try:
        with conn.cursor() as cur:
            if exists and mode == "overwrite":
                cur.execute(f"""
                    DELETE FROM {table_name}
                    WHERE time = %s::timestamptz
                """, (ts,))

            # ON CONFLICT DO NOTHING：唯一约束冲突时跳过该行，不报错
            execute_values(cur, f"""
                INSERT INTO {table_name} (time, city, price_type, price_cny_mwh)
                VALUES %s
                ON CONFLICT (time, city, price_type) DO NOTHING
            """, rows)

        peak_count = len([r for r in rows if r[2] == 'peak'])
        valley_count = len([r for r in rows if r[2] == 'valley'])
        print(f"✅ [DB] 写入成功：{date_str} | 高峰 {peak_count} 条 | 低谷 {valley_count} 条")
        return True
    except Exception as exc:
        print(f"❌ [DB] 日期 {date_str} 写入失败: {exc}")
        return False


def write_to_db(conn, cfg: dict, all_data: list):
    """
    将采集的数据批量写入数据库（保留用于兼容，现已被 write_one_day_to_db 取代）
    all_data: [{"date": "2026-07-01", "peak": [...], "valley": [...]}, ...]
    """
    table_name = cfg['measurement']
    timezone = cfg['timezone']

    # 构建写入数据
    rows = []
    for item in all_data:
        date_str = item['date']
        # 转换日期为带时区的时间戳
        dt = pd.to_datetime(date_str).tz_localize(timezone)

        # 高峰数据
        for peak in item.get('peak', []):
            if peak.get('price'):
                rows.append((
                    dt.to_pydatetime(),
                    peak.get('city', ''),
                    'peak',
                    float(peak['price'])
                ))

        # 低谷数据
        for valley in item.get('valley', []):
            if valley.get('price'):
                rows.append((
                    dt.to_pydatetime(),
                    valley.get('city', ''),
                    'valley',
                    float(valley['price'])
                ))

    if not rows:
        print("⚠️  没有数据可写入数据库")
        return False

    try:
        with conn.cursor() as cur:
            # 先删除已有数据（按日期范围）
            dates = [item['date'] for item in all_data]
            min_date = min(dates)
            max_date = max(dates)
            cur.execute(f"""
                DELETE FROM {table_name}
                WHERE time >= %s::timestamptz AND time <= %s::timestamptz
            """, (min_date, max_date))

            # 批量插入
            execute_values(cur, f"""
                INSERT INTO {table_name} (time, city, price_type, price_cny_mwh)
                VALUES %s
            """, rows)

        print(f"✅ 数据库写入成功：{len(rows)} 条记录（{min_date} ~ {max_date}）")
        return True
    except Exception as exc:
        print(f"❌ 数据库写入失败: {exc}")
        return False


# =====================================================================
#  原有功能（保留）
# =====================================================================

def human_click(page, locator, delay_before=(0.3, 0.8), delay_after=(0.5, 1.2)):
    """
    模拟真人点击：先移动到元素附近，停顿，再精确移动，点击
    """
    # 获取元素位置
    box = locator.bounding_box()
    if not box:
        # 降级为普通点击
        locator.click()
        return

    # 计算中心点（加一点随机偏移，不像机器人总是点正中心）
    center_x = box['x'] + box['width'] / 2 + random.uniform(-3, 3)
    center_y = box['y'] + box['height'] / 2 + random.uniform(-2, 2)

    # 先移动到附近（模拟人眼定位）
    page.mouse.move(center_x - random.randint(30, 60), center_y - random.randint(20, 40))
    random_sleep(*delay_before)

    # 精确移动到目标位置
    page.mouse.move(center_x, center_y)
    random_sleep(0.1, 0.25)

    # 点击（需要传入坐标）
    page.mouse.click(center_x, center_y)
    random_sleep(*delay_after)


def select_calendar_date(page, target_year: int, target_month: int, target_day: int):
    """
    支持跨月份的日期选择：
    1. 打开日历
    2. 检查目标日期是否在当前视图
    3. 如果不在，导航到目标月份
    4. 点击目标日期
    """
    target_date_str = f"{target_year}-{target_month:02d}-{target_day:02d}"

    # 1. 点击日期输入框打开日历（模拟真人）
    picker_input = page.locator('.ant-picker-input input[placeholder="请选择日期"]').first
    print(f"🖱️  点击日期输入框...")
    human_click(page, picker_input, delay_before=(0.5, 1.0), delay_after=(1.0, 2.0))

    # 2. 等待日历弹出（检测日期单元格是否可见）
    try:
        page.locator('td.ant-picker-cell').first.wait_for(state='visible', timeout=5000)
        print("✅ 日历已打开")
    except:
        print("⚠️  日历可能未打开，尝试继续...")

    random_sleep(0.5, 1.0)

    # 3. 检查目标日期是否在当前视图中
    day_cell = page.locator(f'td.ant-picker-cell[title="{target_date_str}"]')

    # 如果不在当前视图，需要导航
    if day_cell.count() == 0:
        print(f"📅 目标日期 {target_date_str} 不在当前视图，需要导航...")

        # 定位月份导航按钮
        prev_month_btn = page.locator('button.ant-picker-header-prev-btn').first
        next_month_btn = page.locator('button.ant-picker-header-next-btn').first

        # 获取当前显示的年月（从 header 读取）
        header_view = page.locator('div.ant-picker-header-view').first
        current_view_text = header_view.text_content()
        print(f"📆 当前日历视图：{current_view_text}")

        # 解析当前显示的年月
        match = re.search(r'(\d{4})年\s*(\d{1,2})月', current_view_text)
        if not match:
            print(f"❌ 无法解析日历视图：{current_view_text}")
            return

        curr_year = int(match.group(1))
        curr_month = int(match.group(2))

        # 导航到目标月份（最多导航 12 次）
        max_nav_attempts = 12
        nav_count = 0

        while nav_count < max_nav_attempts:
            # 再次检查目标日期是否已可见
            day_cell = page.locator(f'td.ant-picker-cell[title="{target_date_str}"]')
            if day_cell.count() > 0:
                print(f"✅ 目标日期已可见")
                break

            # 判断导航方向
            if curr_year > target_year or (curr_year == target_year and curr_month > target_month):
                # 需要往前导航
                print(f"◀️  点击上个月按钮...")
                human_click(page, prev_month_btn, delay_before=(0.3, 0.6), delay_after=(0.8, 1.2))
                curr_month -= 1
                if curr_month < 1:
                    curr_month = 12
                    curr_year -= 1
            else:
                # 需要往后导航
                print(f"▶️  点击下个月按钮...")
                human_click(page, next_month_btn, delay_before=(0.3, 0.6), delay_after=(0.8, 1.2))
                curr_month += 1
                if curr_month > 12:
                    curr_month = 1
                    curr_year += 1

            nav_count += 1
            random_sleep(0.5, 1.0)

        if nav_count >= max_nav_attempts:
            print(f"❌ 导航失败，无法到达目标月份")
            return

    # 4. 再次获取日期单元格（可能已经导航成功）
    day_cell = page.locator(f'td.ant-picker-cell[title="{target_date_str}"]')

    if day_cell.count() == 0:
        print(f"❌ 未找到日期 {target_date_str}")
        return

    # 检查是否可点击（不是 disabled）
    cell_class = day_cell.first.get_attribute('class') or ''
    if 'ant-picker-cell-disabled' in cell_class:
        print(f"⚠️  日期 {target_date_str} 不可选择（已禁用）")
        return

    # 5. 点击日期（模拟真人）
    print(f"🖱️  点击日期 {target_year}-{target_month:02d}-{target_day:02d}...")
    human_click(page, day_cell.first, delay_before=(0.3, 0.6), delay_after=(1.0, 1.8))

    # 6. 等待日历关闭 / 数据加载
    random_sleep(1.0, 2.0)
    print(f"✅ 已选中日期 {target_year}-{target_month:02d}-{target_day:02d}")


def attach_diagnostics(page):
    """给 page 挂载调试监听器（同时把 WAF 拦截 / 数据接口状态记录到 PAGE_STATE）"""
    def on_console(msg):
        if msg.type != "debug":
            print(f"[CONSOLE] [{msg.type}] {msg.text}")

    def on_pageerror(error):
        print(f"[JS_ERROR] {error}")

    def on_requestfailed(request):
        print(f"[NET_FAIL] {request.method} {request.url}  reason={request.failure}")

    def on_response(response):
        status = response.status
        url = response.url
        if status != 200:
            print(f"[HTTP_{status}] {response.request.method} {url}")

        # 记录所有 XHR/fetch URL（供 --probe 定位真正的电价数据接口）
        try:
            if response.request.resource_type in ("xhr", "fetch"):
                PAGE_STATE["seen_urls"].append((status, url))
        except Exception:
            pass

        # 只有当用户显式配置了数据接口关键字时，才据此判断 WAF 拦截。
        # （不能用"任意 /portal/ 4xx"来判定：日志里 marketEntitiesNum 每天都 400，
        #   但它只是首页计数器，与电价数据无关，会造成 reload 误报。）
        if DATA_API_KEYWORD and DATA_API_KEYWORD in url:
            PAGE_STATE["data_api_status"] = status
            if status >= 400:
                PAGE_STATE["waf_block_count"] += 1
                PAGE_STATE["reload_needed"] = True

    page.on("console", on_console)
    page.on("pageerror", on_pageerror)
    page.on("requestfailed", on_requestfailed)
    page.on("response", on_response)


def page_is_ready(page, timeout_ms=8000):
    """严格检查页面 SPA 是否活着：日期选择器必须可见。
    注意：绝不能拿"body 有文字"当就绪信号 —— 反爬 400 挑战页/错误页也有文字，会误判。"""
    try:
        page.locator('.ant-picker-input input[placeholder="请选择日期"]').first.wait_for(
            state="visible", timeout=timeout_ms)
        return True
    except Exception:
        return False


def wait_for_spa_ready(page, timeout_ms=60000):
    """SPA 就绪检测（严格版）：只有日期选择器出现才算就绪"""
    if page_is_ready(page, timeout_ms=timeout_ms):
        print("✅ [SPA_READY] 日期选择器可见，页面就绪！")
        return True
    print("❌ [SPA_READY] 等待超时，日期选择器未出现（页面空白/被反爬拦截？）")
    return False


def ensure_page_alive(page, max_reloads=3, interactive=True):
    """
    确保页面活着（日期选择器可见）。死了就反复 reload 尝试恢复；
    全部失败且 interactive=True 时，暂停等待用户在浏览器里手动救活
    （人工过反爬挑战比任何自动化手段都可靠）。
    返回: True = 页面可用；False = 无法恢复（建议中止本轮）
    """
    if page_is_ready(page, timeout_ms=5000):
        return True

    for i in range(1, max_reloads + 1):
        print(f"⚠️  页面已失活（日期选择器消失），第 {i}/{max_reloads} 次尝试 reload 恢复...")
        try:
            page.reload(wait_until="domcontentloaded", timeout=30000)
        except Exception as e:
            print(f"   reload 异常：{e}")
        # 给反爬挑战留足自动解决的时间
        if page_is_ready(page, timeout_ms=20000):
            print("✅ 页面已恢复")
            reset_page_state()
            random_sleep(2, 3)
            return True
        random_sleep(3, 5)

    if interactive:
        print("=" * 60)
        print("❌ 页面无法自动恢复（大概率被反爬 WAF 持续拦截，显示空白/400）。")
        print("💡 请在弹出的 Edge 窗口里手动处理：")
        print("   1. 地址栏重新输入目标网址并回车")
        print("   2. 如出现反爬挑战/登录页，人工完成它")
        print("   3. 确认页面正常显示日期选择器和地图后，回到本窗口按回车继续")
        print("=" * 60)
        try:
            input()
        except EOFError:
            return False
        return page_is_ready(page, timeout_ms=15000)
    return False


def save_debug_info(page, label="debug"):
    """保存截图 + HTML（全部 best-effort：短超时 + 吞掉异常，
    绝不因为页面卡死/白屏而抛错打断主恢复流程）"""
    try:
        page.screenshot(path=f"{label}_screenshot.png", full_page=False, timeout=6000)
    except Exception as e:
        print(f"⚠️  调试截图失败（已跳过）：{type(e).__name__}")
    try:
        with open(f"{label}_page.html", "w", encoding="utf-8") as f:
            f.write(page.content())
    except Exception as e:
        print(f"⚠️  调试 HTML 保存失败（已跳过）：{type(e).__name__}")
    print(f"📦 调试信息已尝试保存：{label}")


def launch_edge_with_debug():
    """启动带远程调试的 Edge 浏览器"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    profile_dir = os.path.join(script_dir, "edge_debug_profile")

    edge_exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    if not os.path.exists(edge_exe):
        edge_exe = r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"

    if not os.path.exists(edge_exe):
        print("❌ 找不到 Edge 浏览器！")
        return None, profile_dir

    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', DEBUG_PORT))
    sock.close()

    if result == 0:
        print(f"✅ 远程调试端口 {DEBUG_PORT} 已开启，直接连接...")
        return None, profile_dir

    print(f"🚀 正在启动 Edge（调试端口 {DEBUG_PORT}）...")
    proc = subprocess.Popen([
        edge_exe,
        f"--remote-debugging-port={DEBUG_PORT}",
        f"--user-data-dir={profile_dir}",
        "--start-maximized",
        # 隐身参数：去掉 navigator.webdriver 等自动化特征，降低被瑞数识别的概率
        "--disable-blink-features=AutomationControlled",
        "--no-first-run",
        "--no-default-browser-check",
        # 直接以 Edge 原生导航打开目标页：反爬挑战在"真实浏览器导航"下更容易自动通过，
        # 比随后用 Playwright 的 page.goto 触发挑战成功率高
        URL,
    ])
    print("⏳ 等待 Edge 启动并自动通过反爬挑战（约 10 秒，请勿关闭该窗口）...")
    time.sleep(10)
    print("✅ Edge 已启动")
    return proc, profile_dir


# =====================================================================
#  主流程
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  广东电力交易中心 - 地图数据采集")
    print("=" * 60)

    # 是否无头（--headless）：B方案=无头免登录
    HEADLESS = "--headless" in sys.argv

    # --overwrite：数据库中已存在的日期也重新采集并覆盖（用于清理脏数据后重采）
    if "--overwrite" in sys.argv:
        DEDUPE_MODE = "overwrite"
        print("⚠️  覆盖模式（--overwrite）：已存在的日期将删除后重写")

    edge_proc = None
    with sync_playwright() as p:
        if HEADLESS:
            print("\n🔗 无头模式启动（复用 profile 保 cookie，过 WAF）...")
            profile_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "edge_debug_profile")
            try:
                browser = p.chromium.launch_persistent_context(
                    user_data_dir=profile_dir,
                    headless=True, channel="msedge",
                    args=["--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage"])
                print("✅ 无头 Edge(复用profile) 已启动")
            except Exception as e:
                print(f"⚠ msedge持久上下文失败({e})，回退无头Chromium(需playwright install)")
                browser = p.chromium.launch_persistent_context(
                    user_data_dir=profile_dir,
                    headless=True,
                    args=["--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage"])
                print("✅ 无头 Chromium(持久上下文) 已启动")
            page = browser.pages[0] if browser.pages else browser.new_page()
        else:
            edge_proc, profile_dir = launch_edge_with_debug()
            print(f"\n🔗 正在连接到 Edge（端口 {DEBUG_PORT}）...")
            try:
                browser = p.chromium.connect_over_cdp(f"http://127.0.0.1:{DEBUG_PORT}")
            except Exception as e:
                print(f"❌ 连接失败：{e}")
                print("💡 请确保：1) 关闭所有其他 Edge 窗口  2) 重新运行脚本")
                if edge_proc:
                    edge_proc.terminate()
                exit()
            print("✅ 已连接到 Edge！")
            contexts = browser.contexts
            if contexts and contexts[0].pages:
                page = contexts[0].pages[0]
                print(f"📄 使用已有标签页：{page.url}")
            else:
                page = contexts[0].new_page()
                print("📄 创建了新的标签页")

        attach_diagnostics(page)

        try:
            # 若浏览器已在目标站点（脚本启动时直接打开了 URL，或你手动打开并过了反爬），
            # 就跳过 goto —— 重新导航会再次触发瑞数挑战，很可能被拦成 400 白屏
            on_site = False
            try:
                on_site = "pm.gd.csg.cn" in (page.url or "")
            except Exception:
                pass

            if on_site:
                print(f"✅ 浏览器已在目标站点，跳过重新导航，直接等待渲染：{page.url}")
            else:
                print(f"\n🌐 正在访问：{URL}")
                page.goto(URL, wait_until="domcontentloaded", timeout=30000)
                print("📄 DOM 已加载，等待 SPA 渲染...")

            if not wait_for_spa_ready(page, timeout_ms=60000):
                print("⚠️  页面未就绪（白屏/被反爬拦截），尝试自动恢复；"
                      "如失败请在浏览器窗口手动过验证...")
                save_debug_info(page, "before_reload")
                # ensure_page_alive 会 reload 重试，仍不行则暂停等你在 Edge 里人工处理
                if not ensure_page_alive(page, max_reloads=2, interactive=not HEADLESS):
                    print("❌ 页面仍无法加载，退出")
                    save_debug_info(page, "final")
                    browser.close()
                    if edge_proc:
                        edge_proc.terminate()
                    exit()

            print("✅ 页面渲染完成！")

        except Exception as e:
            print(f"❌ 页面加载异常：{e}")
            save_debug_info(page, "error")
            browser.close()
            if edge_proc:
                edge_proc.terminate()
            exit()

        # ========== 开始数据采集 ==========
        # --probe：探测模式，只跑一个日期，打印 ECharts 结构 + 所有 XHR URL 后退出（不写库）
        if "--probe" in sys.argv:
            today = datetime.datetime.now().date()
            run_probe(page, today.year, today.month, today.day)
            browser.close()
            if edge_proc:
                edge_proc.terminate()
            exit()

        # ========== 预先建立数据库连接（每天复用，最后统一关闭）==========
        db_conn = get_db_connection(DB_CONFIG)
        if db_conn:
            ensure_db_table(db_conn, DB_CONFIG)
        else:
            print("❌ 数据库连接失败，无法继续采集")
            browser.close()
            if edge_proc:
                edge_proc.terminate()
            exit()

        # 生成日期列表（支持跨月份）
        # 示例：2026年6月25日 - 2026年7月5日
        start_date = datetime.date(2025, 7, 1)
        # end_date = datetime.date(2026, 7, 20)
        end_date=datetime.datetime.now().date()

        date_list = []
        current_date = start_date
        while current_date <= end_date:
            date_list.append((current_date.year, current_date.month, current_date.day))
            current_date += datetime.timedelta(days=1)

        print(f"\n📅 待处理日期：{start_date} - {end_date}（共 {len(date_list)} 天）")

        # 采集统计 & 跨天数据签名（用于检测"数据冻结"）
        prev_signature = None
        stats = {"ok": 0, "skipped": 0, "invalid": 0, "error": 0}
        invalid_dates = []

        for i, (y, m, d) in enumerate(date_list, 1):
            print(f"\n{'='*60}")
            print(f"🔍 [{i}/{len(date_list)}] 开始处理日期：{y}-{m:02d}-{d:02d}")
            print(f"{'='*60}")
            try:
                # 页面存活检查：上一个日期可能把页面搞死（reload 被 WAF 拦截等），
                # 先救活再继续；救不活就中止本轮，避免剩余日期全部空转报废。
                # 无头模式没有窗口可手动干预，interactive 关闭（直接中止）
                if not ensure_page_alive(page, interactive=not HEADLESS):
                    print("❌ 页面无法恢复（含手动干预），中止本轮采集。"
                          f"剩余 {len(date_list) - i + 1} 个日期未处理，请解决反爬后重新运行。")
                    stats["error"] += len(date_list) - i + 1
                    break

                # 定期主动 reload 页面，刷新反爬 token（防止跑久了被 WAF 拦截）
                if RELOAD_EVERY_N_DATES and i > 1 and (i - 1) % RELOAD_EVERY_N_DATES == 0:
                    print(f"🔁 已处理 {i-1} 个日期，定期 reload 页面刷新 token...")
                    reload_and_wait(page)

                # 每个日期处理前检查一次连接，断开则自动重连
                db_conn = ensure_db_connection(db_conn, DB_CONFIG)
                if db_conn is None:
                    print(f"❌ 数据库重连失败，跳过日期 {y}-{m:02d}-{d:02d}")
                    stats["error"] += 1
                    continue

                result = process_one_date(
                    page, y, m, d,
                    db_conn=db_conn, db_cfg=DB_CONFIG,
                    prev_signature=prev_signature,
                )
                # 把（可能已重连的）连接拿回来，下一轮循环用新的
                if result and result.get('_db_conn') is not None:
                    db_conn = result['_db_conn']

                # 统计 & 更新跨天签名
                if result and result.get("skipped"):
                    stats["skipped"] += 1
                elif result and result.get("valid"):
                    stats["ok"] += 1
                    prev_signature = result.get("signature", prev_signature)
                    print(f"✅ 日期 {y}-{m:02d}-{d:02d} 处理完成并写库")
                else:
                    stats["invalid"] += 1
                    invalid_dates.append(f"{y}-{m:02d}-{d:02d}")
                    print(f"⚠️  日期 {y}-{m:02d}-{d:02d} 数据不可信，未写库")
            except Exception as e:
                print(f"❌ 处理失败：{e}")
                stats["error"] += 1
                import traceback
                traceback.print_exc()

        # ========== 关闭数据库连接 ==========
        db_conn.close()
        print("\n✅ 数据库连接已关闭")

        print("\n" + "=" * 60)
        print("✅ 所有日期处理完成！")
        print(f"📊 统计：成功写库 {stats['ok']} | 已存在跳过 {stats['skipped']} "
              f"| 数据不可信 {stats['invalid']} | 异常 {stats['error']}")
        if invalid_dates:
            print(f"⚠️  以下日期数据不可信（未写库），建议排查后重采：")
            for ds in invalid_dates:
                print(f"     - {ds}")
        print(f"🗄️  数据库：{DB_CONFIG['ts_dbname']}.{DB_CONFIG['measurement']}")
        print("=" * 60)
        input()
        browser.close()
        if edge_proc:
            edge_proc.terminate()
