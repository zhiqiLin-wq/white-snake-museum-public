"""D6 一致性评测集生成器：30 条用户偏好 × 3 种不同措辞。

让 LLM 把每条偏好改写成 3 种语义等价、措辞不同的用户陈述，
用于评测记忆系统对同一偏好的提取稳定性（一致性）。
30 条偏好覆盖：研究偏好/身份、约束/否定、观点/评价、交流风格/格式。
输出 tests/eval/data/d6_consistency.json
"""

import asyncio
import json
import sys
from pathlib import Path

_eval_dir = Path(__file__).resolve().parent
_project_root = _eval_dir.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_eval_dir))

from common import write_json
from agent.server.llm.deepseek_provider import DeepSeekProvider
from agent.server.config import settings

MODEL = settings.memory_extraction_model or settings.effective_fast_model

PREFERENCES = (
    # 研究偏好 / 身份（10 条，原有）
    ["研究方向是宋代江南民间信仰与文学互动",
     "最喜欢的文学人物是法海",
     "不喜欢收到长篇回复",
     "偏好文本细读与宏观历史分析相结合",
     "最关注《西湖三塔记》的三怪母题",
     "认为《白娘子永镇雷峰塔》是白蛇传定型的标志",
     "研究时要求引用文献原文并标注出处",
     "偏好用母题流变视角分析版本演变",
     "关注雷峰塔意象从镇妖到镇人的变迁",
     "认为田汉京剧版是白蛇传现代化的重要节点"]
    # 约束 / 否定类（7 条）
    + ["不喜欢把白蛇传和现代恋爱剧混为一谈",
       "反感过度解读文献、硬套理论",
       "不喜欢堆砌版本罗列而没有分析",
       "要求引文准确、不能凭记忆转述",
       "不喜欢用现代价值观评判古代作品",
       "不喜欢回复里出现英文术语",
       "讨论版本差异时必须给出年代"]
    # 观点 / 评价类（7 条）
    + ["认为法海有卫道者和悲剧推动者的双重性",
       "认为白娘子经历了从妖性到人性的转变",
       "认为田汉京剧改编强化了反封建主题",
       "觉得小青从青鱼精到青蛇反映了民间想象的演变",
       "认为《白娘子永镇雷峰塔》悲剧结局比大团圆更有文学价值",
       "认为许仙在不同版本中形象差异很大",
       "觉得白蛇传与梁祝孟姜女并称的提法值得商榷"]
    # 交流风格 / 格式类（6 条）
    + ["喜欢先给结论再展开分析",
       "回答时喜欢分点列出",
       "喜欢回答带具体例子",
       "先读原始文献再看二手研究",
       "把问题放到具体历史语境里理解",
       "引用古籍时要标明卷次"]
)

REWRITE_USER = (
    "请把下面的用户偏好改写成 3 种不同措辞的自然口语陈述。\n"
    "要求：\n"
    "1. 每条都是用户描述自己长期习惯/偏好的陈述句（如\"我习惯先给结论再分析\"\"我比较喜欢分点看\"），"
    "而不是对当前对话发出的祈使命令（不要写成\"你能不能先说结论\"\"你分点说\"）。\n"
    "2. 每种措辞语义必须严格等价于原偏好，但用词、句式要明显不同（不要只是换个同义词）。\n"
    "偏好：{preference}\n"
    "只输出 JSON 数组，元素格式：[\"措辞1\", \"措辞2\", \"措辞3\"]"
)

GENERATOR_SYSTEM = "你是一个严谨的数据集生成器，只输出合法 JSON，不输出解释性文字。"


async def main():
    llm = DeepSeekProvider()
    prefs = []
    for i, pref in enumerate(PREFERENCES, 1):
        user = REWRITE_USER.format(preference=pref)
        data = None
        for _ in range(5):
            try:
                resp = await llm.generate(system=GENERATOR_SYSTEM, user=user, model=MODEL,
                                          max_tokens=512, temperature=0.8)
                import re
                txt = resp.content.strip()
                m = re.search(r"\[[\s\S]*\]", txt)
                if m:
                    data = json.loads(m.group(0))
                else:
                    data = json.loads(txt)
                break
            except Exception:
                data = None
                await asyncio.sleep(3)
        if not isinstance(data, list) or len(data) < 3:
            print(f"  [警告] 偏好 {i} 生成失败，跳过")
            continue
        prefs.append({
            "id": f"p{i:02d}",
            "preference": pref,
            "variants": [str(v).strip() for v in data[:3]],
        })
        print(f"  [偏好 {i}/{len(PREFERENCES)}] {pref[:20]}... -> {len(data[:3])} 种措辞")
        await asyncio.sleep(1)

    out = {"meta": {"name": "D6 一致性评测集", "total_preferences": len(prefs),
                    "variants_per_preference": 3}, "preferences": prefs}
    path = write_json(out, "d6_consistency.json")
    print(f"\n完成：{len(prefs)} 条偏好。写入 {path}")


if __name__ == "__main__":
    asyncio.run(main())
