"""D2 提取评测集生成器：60 条金标准事实 + LLM 生成用户陈述。

60 条金标准事实（Entity 20 / Archival 20 / System 20，结合白蛇传领域），
其中 Entity 含 10 条身份/偏好 + 10 条观点/评价。
每条让 LLM 生成一段自然的用户陈述（语义等价）。评测提取 P/R/F1 与分类准确率。
输出 tests/eval/data/d2_extraction.json
"""

import asyncio
import json
import re
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

GOLD_FACTS = (
    # Entity: 身份 / 偏好（10 条）
    [("Entity", "用户的名字是小研"),
     ("Entity", "用户的研究方向是宋代江南民间信仰与文学互动"),
     ("Entity", "用户最喜欢的文学人物是法海"),
     ("Entity", "用户偏好文本细读与宏观历史分析相结合"),
     ("Entity", "用户最关注《西湖三塔记》的三怪母题"),
     ("Entity", "用户认为《白娘子永镇雷峰塔》是白蛇传定型的标志"),
     ("Entity", "用户偏好用母题流变视角分析版本演变"),
     ("Entity", "用户关注雷峰塔意象从镇妖到镇人的变迁"),
     ("Entity", "用户认为田汉京剧版是白蛇传现代化的重要节点"),
     ("Entity", "用户是专门研究白蛇传文献的学者")]
    # Entity: 观点 / 评价（10 条）
    + [("Entity", "用户认为法海形象有卫道者和悲剧推动者的双重性"),
       ("Entity", "用户觉得白娘子从《西湖三塔记》到《白娘子永镇雷峰塔》经历了从妖性到人性的转变"),
       ("Entity", "用户认为小青从青鱼精演变为青蛇反映了民间想象的流变"),
       ("Entity", "用户评价许仙形象在不同版本中从懦弱到重情的转变"),
       ("Entity", "用户认为雷峰塔倒塌象征旧秩序的解体"),
       ("Entity", "用户觉得水漫金山情节凸显了白娘子的反抗性"),
       ("Entity", "用户认为《白娘子永镇雷峰塔》的悲剧结局比大团圆结局更有文学价值"),
       ("Entity", "用户看重版本年代考订在母题流变研究中的作用"),
       ("Entity", "用户主张研究白蛇传必须回到宋代话本源头"),
       ("Entity", "用户认为田汉京剧版强化了反封建主题")]
    # Archival: 客观事实（20 条）
    + [("Archival", "《太平广记》卷四五八《李黄》中李黄遇白衣女子后化水而亡"),
       ("Archival", "《西湖三塔记》中奚真人将白蛇、乌鸡、獭三怪镇压于三石塔"),
       ("Archival", "《白娘子永镇雷峰塔》是冯梦龙《警世通言》所收的拟话本"),
       ("Archival", "方成培《雷峰塔传奇》是清乾隆时期的戏曲"),
       ("Archival", "陈遇乾《义妖传》是苏州弹词"),
       ("Archival", "田汉《白蛇传》是京剧"),
       ("Archival", "白蛇传核心人物包括白娘子、许仙、法海、小青"),
       ("Archival", "雷峰塔位于杭州西湖边"),
       ("Archival", "金山寺是白蛇传中水漫金山情节的发生地"),
       ("Archival", "玉山堂主人《雷峰塔传奇》是白话神魔小说"),
       ("Archival", "《李黄》出自《太平广记》卷四五八"),
       ("Archival", "白蛇传最早的书面记载出现在宋代话本"),
       ("Archival", "《白娘子永镇雷峰塔》中的许仙原名许宣"),
       ("Archival", "雷峰塔于1924年倒塌"),
       ("Archival", "白蛇传于2006年列入首批国家级非物质文化遗产名录"),
       ("Archival", "端午节喝雄黄酒是白娘子现出蛇形的关键情节"),
       ("Archival", "田汉京剧《白蛇传》改编于二十世纪五十年代"),
       ("Archival", "白蛇传与牛郎织女、孟姜女、梁祝并称中国四大民间传说"),
       ("Archival", "《西湖游览志》是明代田汝成所著"),
       ("Archival", "白蛇传故事通常设定在南宋绍兴年间")]
    # System: 约束（20 条）
    + [("System", "不要发送长代码"),
       ("System", "引用文献必须标注出处"),
       ("System", "不要用现代改编版替代原始文献"),
       ("System", "分析版本演变必须按朝代顺序"),
       ("System", "讨论白蛇传时不要跑题到无关领域"),
       ("System", "回答要简洁"),
       ("System", "涉及不确定的信息要标注待考"),
       ("System", "不要提供编程建议"),
       ("System", "不要提供医疗、法律、金融建议"),
       ("System", "引用原文时要注明版本"),
       ("System", "回答时必须分点列出"),
       ("System", "不要用英文术语替代中文术语"),
       ("System", "引用古籍时要标明卷次"),
       ("System", "讨论版本差异时要给出年代"),
       ("System", "不要过度解读文献"),
       ("System", "涉及争议观点时要说明学术分歧"),
       ("System", "回答字数控制在三百字以内"),
       ("System", "优先使用原始文献而非二手研究"),
       ("System", "不要使用表情符号"),
       ("System", "引文必须准确，不能凭记忆转述")]
)

REWRITE_USER = (
    "请把下面的用户事实改写成一段自然的用户陈述（用户会这样说的话），语义严格等价但措辞自然口语化。\n"
    "事实：{fact}\n"
    "只输出一段用户陈述文本，不要输出 JSON 或解释。"
)

GENERATOR_SYSTEM = "你是一个数据集生成器，把事实改写成自然口语。"


async def main():
    llm = DeepSeekProvider()
    items = []
    for i, (mtype, fact) in enumerate(GOLD_FACTS, 1):
        user = REWRITE_USER.format(fact=fact)
        statement = ""
        for _ in range(5):
            try:
                resp = await llm.generate(system=GENERATOR_SYSTEM, user=user, model=MODEL,
                                          max_tokens=256, temperature=0.8)
                statement = (resp.content or "").strip()
                if statement:
                    break
            except Exception:
                statement = ""
                await asyncio.sleep(3)
        if not statement:
            print(f"  [警告] 事实 {i} 生成失败，跳过")
            continue
        items.append({"id": f"f{i:02d}", "memory_type": mtype,
                      "gold_fact": fact, "statement": statement})
        print(f"  [{i}/{len(GOLD_FACTS)}] {mtype:9s} {fact[:20]}...")
        await asyncio.sleep(0.5)

    out = {"meta": {"name": "D2 提取评测集", "total": len(items),
                    "type_breakdown": {
                        t: sum(1 for x in items if x["memory_type"] == t)
                        for t in ("Entity", "Archival", "System")
                    }},
           "facts": items}
    path = write_json(out, "d2_extraction.json")
    print(f"\n完成：{len(items)} 条。写入 {path}")


if __name__ == "__main__":
    asyncio.run(main())
