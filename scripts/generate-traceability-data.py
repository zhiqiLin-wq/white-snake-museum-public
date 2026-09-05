"""预生成景观-文本溯源数据脚本

离线工具，不在应用运行时执行。一次性运行，为全部 11 个地点生成
mutualConstruction 和 descriptionStyles，合并到 location-literary-data.json。

用法:
  1. 确保 agent/venv 已安装依赖（langchain, anthropic 等）
  2. 设置环境变量 ANTHROPIC_API_KEY
  3. cd scripts && ..\\agent\\venv\\Scripts\\python generate-traceability-data.py

工作流:
  1. 读取 client/src/data/landscape-info.json（物理景观）
  2. 读取 client/src/data/location-literary-data.json（已有 literaryRecords）
  3. 对每个地点构建 LLM prompt
  4. 调用 LLM 生成 mutualConstruction 和 descriptionStyles
  5. 合并写入 location-literary-data.json
  6. 人工审核生成结果
"""

import json
import os
import sys
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent
CLIENT_DATA = ROOT / "client" / "src" / "data"
LANDSCAPE_INFO_PATH = CLIENT_DATA / "landscape-info.json"
LITERARY_DATA_PATH = CLIENT_DATA / "location-literary-data.json"

# 所有需要生成数据的地点（与前端 landscape-info.json 一致）
ALL_LOCATIONS = [
    "西湖", "雷峰塔", "金山寺", "灵隐寺", "望江楼",
    "青城山", "峨眉山", "承天寺", "龙虎山", "卧佛寺", "断桥",
]

SYSTEM_PROMPT = """你是白蛇传景观-文本互构分析专家。分析指定地点的物理景观与文学文本叙事之间的双向影响关系。

## 任务
1. landscapeToText: 分析该地点的物理景观特征（地理/建筑/历史变迁）如何影响了文本中的叙事方式
   （如：景观的视觉特征如何塑造了场景描写？建筑空间的布局如何影响了情节走向？）80-120字。
2. textToLandscape: 分析文本叙事如何反过来塑造了该地点的文化景观认知和社会地位
   （如：文学作品如何改变人们对这个地方的想象？叙事如何赋予建筑以象征意义？）80-120字。
3. descriptionStyles: 为每个朝代生成该地点在文本中的叙事特征描述（每朝代15-25字）。
   若该朝代无文献记录，描述为"XX代文献中未见（该地点名）在白蛇叙事中的记载"。

## 输出格式
只返回 JSON:
{"mutualConstruction": {"landscapeToText": "...", "textToLandscape": "..."},
 "descriptionStyles": {"唐": "...", "宋": "...", "明": "...", "清": "..."}}

注意: 若物理景观数据缺失，请根据文献记录推断景观特征。"""


def load_json(path: Path) -> dict | list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_location_prompt(location_name: str, landscape_info: dict, literary_data: dict) -> str:
    """为单个地点构建用户 prompt。"""
    loc_info = literary_data.get(location_name, {})
    records = loc_info.get("literaryRecords", [])

    # 构建物理景观描述
    phys = landscape_info.get(location_name, {})
    if phys:
        landscape_desc = (
            f"历史名称: {phys.get('historicalName', '')}\n"
            f"今名: {phys.get('currentName', '')}\n"
            f"始建: {phys.get('builtYear', '未知')}\n"
            f"历史变迁: {phys.get('historicalChanges', '')}"
        )
    else:
        landscape_desc = "无物理景观数据"

    # 构建文献记录摘要
    records_lines = []
    for r in records:
        dynasty = r.get("dynasty", "?")
        ch_title = r.get("chapterTitle", "")
        excerpt = r.get("excerpt", "")
        style = r.get("descriptionStyle", "")
        records_lines.append(f"[{dynasty}] {ch_title}: ...{excerpt}... ({style})")

    records_summary = "\n".join(records_lines[:30]) if records_lines else "无文献记录"

    return (
        f"## 地点: {location_name}\n\n"
        f"## 物理景观数据\n{landscape_desc}\n\n"
        f"## 文献记录\n{records_summary}\n\n"
        "请分析该地点的景观-文本互构关系。"
    )


def generate_with_anthropic(system: str, user: str, api_key: str) -> dict:
    """通过 Anthropic API 生成互构分析。"""
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        temperature=0.4,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    content_raw = resp.content[0].text.strip()
    if content_raw.startswith("```"):
        content_raw = content_raw.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(content_raw)


def generate_with_langchain(system: str, user: str) -> dict:
    """通过项目现有的 LLM 基础设施生成。需要从 agent/server 引入。"""
    sys.path.insert(0, str(ROOT / "agent" / "server"))
    from config import settings
    from llm.provider import get_llm

    llm = get_llm()
    model = settings.effective_fast_model
    resp = llm.generate(system, user, model=model, max_tokens=1024, temperature=0.4)
    content_raw = resp.content.strip()
    if content_raw.startswith("```"):
        content_raw = content_raw.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(content_raw)


def main():
    print("=== 景观-文本溯源数据预生成脚本 ===")

    # 1. 加载数据
    if not LANDSCAPE_INFO_PATH.exists():
        print(f"[ERROR] 未找到 landscape-info.json: {LANDSCAPE_INFO_PATH}")
        sys.exit(1)
    if not LITERARY_DATA_PATH.exists():
        print(f"[ERROR] 未找到 location-literary-data.json: {LITERARY_DATA_PATH}")
        sys.exit(1)

    landscape_info_list = load_json(LANDSCAPE_INFO_PATH)
    literary_data = load_json(LITERARY_DATA_PATH)

    # 将 landscape-info 数组转为 dict（key = name）
    landscape_info_dict = {item["name"]: item for item in landscape_info_list}

    print(f"[OK] 已加载 {len(landscape_info_dict)} 个地点的物理景观数据")
    print(f"[OK] 已加载 {len(literary_data)} 个地点的文献溯源数据")

    # 2. 检查哪些地点需要生成
    to_generate = []
    for loc_name in ALL_LOCATIONS:
        loc_data = literary_data.get(loc_name, {})
        has_mutual = bool(loc_data.get("mutualConstruction"))
        has_styles = bool(loc_data.get("descriptionStyles"))
        if has_mutual and has_styles:
            print(f"[SKIP] {loc_name}: 已有完整数据")
        else:
            to_generate.append(loc_name)
            missing = []
            if not has_mutual:
                missing.append("mutualConstruction")
            if not has_styles:
                missing.append("descriptionStyles")
            print(f"[TODO] {loc_name}: 缺少 {', '.join(missing)}")

    if not to_generate:
        print("\n所有地点数据已完整，无需生成。")
        return

    print(f"\n共 {len(to_generate)} 个地点需要生成数据")

    # 3. 选择生成方式
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    use_anthropic = bool(api_key)

    if use_anthropic:
        print("[INFO] 使用 Anthropic API 直接调用")
    else:
        print("[INFO] 使用项目 LangChain LLM (需在 agent/venv 中运行)")

    # 4. 逐地点生成
    for loc_name in to_generate:
        print(f"\n--- 正在生成: {loc_name} ---")

        user_prompt = build_location_prompt(
            loc_name, landscape_info_dict, literary_data
        )

        try:
            if use_anthropic:
                result = generate_with_anthropic(SYSTEM_PROMPT, user_prompt, api_key)
            else:
                result = generate_with_langchain(SYSTEM_PROMPT, user_prompt)

            # 验证结果包含必要字段
            mc = result.get("mutualConstruction", {})
            ds = result.get("descriptionStyles", {})

            if not mc.get("landscapeToText") or not mc.get("textToLandscape"):
                print(f"  [WARN] mutualConstruction 不完整，跳过")
                continue

            if not all(d in ds for d in ["唐", "宋", "明", "清"]):
                print(f"  [WARN] descriptionStyles 不完整，跳过")
                continue

            # 写入 literary_data
            if loc_name not in literary_data:
                literary_data[loc_name] = {
                    "literaryRecords": [],
                    "totalMentions": {"唐": 0, "宋": 0, "明": 0, "清": 0},
                }

            literary_data[loc_name]["mutualConstruction"] = mc
            literary_data[loc_name]["descriptionStyles"] = ds

            print(f"  [OK] 生成成功")

        except Exception as e:
            print(f"  [ERROR] 生成失败: {e}")
            print(f"  继续处理下一个地点...")
            continue

    # 5. 写入文件
    output = json.dumps(literary_data, ensure_ascii=False, indent=2)
    with open(LITERARY_DATA_PATH, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"\n=== 完成 === 数据已写入 {LITERARY_DATA_PATH}")
    print("请人工审核生成内容后再提交。")


if __name__ == "__main__":
    main()
