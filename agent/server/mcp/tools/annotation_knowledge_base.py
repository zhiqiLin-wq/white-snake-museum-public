"""标注领域知识库 — 白蛇传已知实体、类别规则、颜色映射。

阶段 0 — P0-05.
从 annotate_passage.py 提取，供新旧标注管线共享。

词表来源：《白蛇传文献选集全面提取报告.docx》提取的人物/地点/专业术语/事件句子/主题句子。

⚠️ 同时作为 MCP 工具 `annotation_knowledge_base` 暴露：
  - LLM 可查询标注规范、类别说明、已知实体同义词、别名与示例
  - 供 LLM 在不确定"一个词该标 person 还是 term"、"某条规则要求是什么"时查阅，保证标注口径一致
"""

_CATEGORY_REVERSE_MAP: dict[str, str] = {
    "人物名称": "person", "地点名称": "location",
    "事件名称": "event", "专有术语": "term", "叙事母题": "motif",
    "人物": "person", "地点": "location",
    "事件": "event", "术语": "term", "母题": "motif",
    "人名": "person", "专业术语": "term", "事件句子": "event", "主题句子": "motif",
    "person": "person", "location": "location",
    "event": "event", "term": "term", "motif": "motif",
}

DEFAULT_CATEGORY_COLORS = {
    "person": "#68A0C8",
    "location": "#78B898",
    "event": "#68B0B8",
    "term": "#9888C0",
    "motif": "#C898B0",
    "custom": "#B8B878",
}

_ENTITY_KB: dict[str, dict] = {
    "person": {
        "label": "人名",
        "known": [
            "许宣", "小青", "素贞", "白素贞", "法海", "许仙",
            "白娘子", "青儿", "许官人", "白蛇", "青青", "李募事",
            "王主人", "海禅师", "卯奴", "法海禅师", "知县", "金母",
            "李员外", "马日丞", "奚宣赞", "蒋和", "青蛇", "娇容",
            "李克用", "李将仕", "王端", "观音菩萨", "许汉文", "何立",
            "王员外", "吴员外", "周将仕", "王公", "白云仙姑", "赵主管",
            "张主管", "李公甫", "白珍娘", "铁头", "戴先生", "蕊芝",
            "奚真人", "袁子明", "许小乙", "黑风仙", "奚统制", "李黄",
            "法明", "伏虎尊者", "吴人杰", "李听", "白衣之姨", "白衣姝",
            "真武北极大帝", "蕙芝仙子", "袁氏之女", "许晋贤", "郑氏女", "释迦文佛",
            "金慈圣母", "降龙尊者", "韩大尹", "黑道家", "乌鸡婆", "南山老母",
            "四圣观道士", "奚妈妈", "扫地仙", "李琯", "獭精", "王凤山",
            "王明", "白蛇娘娘", "西池金母", "骊山老母",
        ],
        "rules": (
            "提取文中出现的所有人物（包括姓名、字号、称谓、身份，如'侍者''使者''娘子''姨''二女奴'）。只要原文中出现人物就标注，不要遗漏。不标代词(他/她/其/吾/余)。不标单个姓氏字(许/白/李)。"
        ),
        "examples": [
            ("许宣道：小生住在过军桥黑珠巷。", [
                {"entity": "许宣", "category": "person", "explanation": "白蛇传男主角"},
            ]),
        ],
    },
    "location": {
        "label": "地点",
        "known": [
            "西湖", "杭州", "钱塘", "镇江", "金山寺", "苏州",
            "临安", "雷峰塔", "南海", "断桥", "钱塘县", "清风洞",
            "瑶池", "双茶坊巷", "吴家巷", "净慈寺", "西池", "临安府",
            "四圣观", "涌金门", "钱塘门", "双茶巷", "吉利桥", "青城山",
            "裘王府", "官巷口", "承天寺", "清波门", "仇王府花园", "孤山路",
            "峨嵋山", "昭庆寺", "杭州府", "赤山埠", "金池", "针子桥",
            "丰乐楼", "丹霞", "五条巷", "保叔塔寺", "六桥", "坝头",
            "白马庙", "秀王府", "箭桥双茶坊巷", "胥江马日", "苏堤", "蟠桃园",
            "西宁桥", "钱塘江口", "龙虎山", "三潭", "东市", "东新桥",
            "井亭桥", "保俶塔", "保和堂", "六一泉", "冷泉亭", "凤翔",
            "周将仕典当库", "奉诚园", "安化门外", "安邑里", "寿安坊", "嵩山",
            "广寒宫", "庄严寺", "怀青巷口", "放生碑", "昆仑山", "杭州西湖",
            "武当仙山", "水磨头", "永宁里", "沈公井巷", "清河街", "灵隐山",
            "白公堤", "石函桥", "紫府", "紫极霄", "花市街", "苏公堤",
            "茭白荡", "过军桥黑珠巷", "连环洞", "金牛寺", "长安", "长安东市",
            "闾门", "真武北极大帝朝拜天阙处", "金华将军庙",
        ],
        "rules": (
            "提取文中提及的地理位置、建筑或场所。不标泛指(山中/寺里/城门/街上/巷口)。不标单个通用字(山/寺/门/桥/塔/楼)，除非它是专名的一部分如'断桥'。"
        ),
        "examples": [
            ("许宣出涌金门，到西湖边。", [
                {"entity": "涌金门", "category": "location", "explanation": "杭州城门"},
            ]),
        ],
    },
    "term": {
        "label": "专业术语",
        "known": [
            "西皮", "白蛇", "前腔", "散板", "西皮散板", "钵盂",
            "主管", "雷峰塔", "原形", "大尹", "神将", "云头",
            "生药", "端阳", "金母", "出首", "马日丞", "尾声",
            "瑶池", "檀香", "西池", "雄黄酒", "倒板", "弥陀",
            "皈依", "真言", "结亲", "袈裟", "中吕过曲", "醒酒汤",
            "青城山", "黄道吉日", "保叔塔", "垛板", "符水", "蟠桃",
            "西皮垛板", "誓愿", "业畜", "五鬼", "仇王府", "募化",
            "午时", "南梆子", "发配", "寿诞", "斗法", "柳叶",
            "点绛唇", "话本", "轮回", "铁笼", "黑气", "七月身孕",
            "保状", "拜堂", "牢城营", "犊车", "琼浆", "红罗帐",
            "缉捕使臣", "花烛", "表字", "金池", "长解", "上苑",
            "丹田", "丹霞", "传奇", "佛会", "县役", "吞吃",
            "园林好", "天下乐", "将军柱", "川拨棹", "恩将仇报", "懒画眉",
            "投坛状", "散符", "泣颜回", "泥丸", "洞房", "浪淘沙",
            "浮图", "真元", "聘礼", "胥江马日", "般若", "金珠细软",
            "钱塘县库银", "除服", "龙杖", "龙虎山", "一夜夫妻百日恩", "七十二小洞",
            "万佛", "三车", "临江仙", "乌鸡", "九转功", "五雷天心正法",
            "京剧", "伏虎尊者", "保和堂", "元母忌日", "八大洞", "八节",
            "化化轮回重化化", "十世修来同船渡", "十里亭", "千里姻缘一线牵", "南廊阁子库募事官", "卯胎",
            "吴越", "哪吒令", "回文", "坎水", "坐化", "天阙",
            "孽菩提", "安魂定魄丹", "寄生草", "小安板", "尚书", "岳相公",
            "广寒宫", "开光", "徒罪", "心猿意马", "忒忒令", "戏术",
            "抹额", "拈花笑", "拟话本", "换头", "昆仑山", "柏亭",
            "桂枝香", "毗岚", "沁园春", "沉醉东风", "油葫芦", "浴佛",
            "火号", "火轮车", "煞尾", "王母", "生生转变再生生", "白话神魔小说",
            "百世修来共枕眠", "皂罗袍", "盐铁使", "真武北极大帝", "石塔", "禅院",
            "紫府", "纸马", "经幡", "统制官", "绣带儿", "羽调 望吾乡",
            "舍利", "色即是空空即色", "苏州弹词", "药行", "蕙芝仙子", "蚩虎",
            "行枷", "衲衣", "褊衫", "西皮操板", "角黍", "训迪教诲",
            "调选", "贝多", "踏破铁鞋无觅处", "醉太平", "醍醐", "释迦佛",
            "金刚体", "金吾参军", "钱垛", "铁树开花始见春", "银装", "锡杖",
            "锦帐", "阎王判你三更到", "阴阳气", "降香", "降龙尊者", "风幡动",
            "香柳娘", "骨塔", "高贤", "鹊踏枝", "黄荆", "黄钟 降黄龙",
            "黑罡风", "九转功夫", "人无害虎心虎有伤人意", "仙吕·忆帝京", "十八罗汉", "座下",
            "步罡踏斗", "浮图/宝塔", "滚绣球", "西湖水干江潮不起", "逢山开路遇水搭桥", "雷峰塔倒白蛇出世",
            "骊山老母",
        ],
        "rules": (
            "提取文中涉及的古代职官、典章制度、特定称谓或文言特殊词汇。不标日常常用词(道/佛/经/庙/缘)。术语可能为2-4字的固定搭配。"
        ),
        "examples": [
            ("法海将钵盂一罩，白娘子现了原形。", [
                {"entity": "钵盂", "category": "term", "explanation": "法海降妖法器"},
            ]),
        ],
    },
    "event": {
        "label": "事件句子",
        "known": [
            "李黄东市遇白衣姝", "白衣姝借宿李家", "姨母接待", "李黄三日欢饮后身消", "李黄归家身消头存", "皂荚树十五千",
            "巨白蛇传闻", "李琯遇二女奴", "李琯脑裂而卒", "枯槐蛇迹", "奚宣赞清明游西湖", "宣赞遇迷途白衣女",
            "卯奴被带回奚家", "婆婆寻女到奚家", "白衣妇人现身", "取心肝吓宣赞", "卯奴救宣赞", "宣赞被放回",
            "奚真人施法捉妖", "三怪现形被镇", "西湖景致介绍", "许宣保叔塔烧香遇雨", "白娘子搭船", "许宣借伞",
            "讨伞遇青青", "白娘子赠银求亲", "五十两银子", "银子事发许宣被捕", "白娘子现银遁走", "许宣配苏州",
            "白娘子追至苏州", "许宣题诗", "卧佛寺遇道士", "白娘子斗法", "许宣再被捉", "许宣配镇江",
            "白娘子寻至镇江", "李克用见色起意", "李员外见蛇惊倒", "金山寺遇法海", "白蛇跳江", "钵盂收妖",
            "白蛇现形", "雷峰塔镇蛇", "法海偈语", "许宣出家坐化", "开宗点题", "付钵授法",
            "法海受命", "黑风仙劝妹", "白蛇下山", "黑风仙苦劝", "许宣清明扫墓", "青青自述",
            "白素贞收青", "斗法收青", "舟遇许宣", "白素贞施法起雨", "船上借伞", "白素贞诉身世",
            "订盟结亲", "赠银定情", "端午查白", "金山水斗", "断桥相会", "炼塔镇妖",
            "塔叙团圆", "祭塔超度", "白蛇降凡", "大帝阻路", "白蛇发誓", "许仙入药店",
            "清明扫墓遇二美", "雨中搭船", "借伞定情", "白珍娘赠银", "李公甫认银出首", "许仙受刑",
            "公甫作证", "许仙招供", "知县发配", "王员外送书", "白珍娘千里寻夫", "白珍娘旅店成亲",
            "法海示妖", "游金山法海示妖", "素贞扫叶", "金母降临", "老母劝金母度妖", "素贞被贬",
            "金母赐琼浆", "素贞下凡", "消除毒气", "遇黑风仙", "白素贞游湖", "断桥借伞",
            "船上巧遇", "依依分别", "结亲成婚", "法海查访", "法海探店", "许仙不信",
            "端阳劝酒", "白蛇饮药现形", "许仙悔悟",
        ],
        "rules": (
            "提取直接推动故事情节发展、包含明确动作或人物交互的客观叙事句子。事件句子的实体原文必须一字不差引用原文完整原句，禁止删改字词、禁止用自己的话概括、禁止加标点断句。并在概括词列给出该事件的概括名（如'端午惊变''水漫金山'）。"
        ),
        "examples": [
            ("许宣祭塔，白素贞得超度。", [
                {"entity": "许宣祭塔，白素贞得超度。", "category": "event", "explanation": "祭塔超度"},
            ]),
            ("雷峰塔中白素贞与许宣相见。", [
                {"entity": "雷峰塔中白素贞与许宣相见。", "category": "event", "explanation": "塔叙团圆"},
            ]),
            ("法海将白素贞镇压于雷峰塔下。", [
                {"entity": "法海将白素贞镇压于雷峰塔下。", "category": "event", "explanation": "炼塔镇妖"},
            ]),
        ],
    },
    "motif": {
        "label": "主题句子",
        "known": [
            "法海劝世偈", "许宣坐化诗", "释迦文佛训诫", "白蛇自述", "法海收妖偈", "黑风仙劝白蛇",
            "金母训素贞", "白蛇发誓", "许宣题诗叹姻缘", "南山老母劝金母", "许仙酒变悔悟", "白素贞端阳叹",
            "小青端阳念", "法海叹世人", "白蛇斗法自辩", "许仙对白素贞", "西湖美景诗", "清明诗",
            "法海题诗劝世", "许宣狱中诗", "西湖赞", "西湖四时景", "西湖真景", "西湖晨昏",
            "西湖物产", "西湖画船", "西湖极乐", "西湖不深不浅", "西湖丽日", "西湖异香",
            "西湖风浪", "白蛇修行自述", "白素贞下山志", "素贞扫叶自述", "素贞修行功行", "素贞采药",
            "素贞自叹", "素贞遇师", "素贞被贬", "金母赐琼浆", "素贞消除毒气", "素贞下凡誓愿",
            "金母指示", "素贞拜别", "金母点化", "白蛇千年修炼", "素贞感恩", "白素贞诉情",
            "白娘子追夫", "白素贞断桥叹", "白素贞游湖", "白素贞感夫恩", "白素贞劝酒叹", "白素贞饮雄黄",
            "许仙悔劝酒", "白素贞离湖", "许仙叹白娘子", "白素贞订盟", "白素贞赠银", "白素贞诉苦",
            "法海收妖", "法海降妖", "钵盂收妖", "白蛇现形", "青鱼现形", "白蛇原形",
            "法海镇塔", "法海点化许宣", "释迦文佛旨意", "法海斗法", "法海金山寺", "法海度许仙",
            "法海警告", "法海端阳计", "法海偈语", "青青自述", "青青斗法", "青青被收",
            "青青侍奉", "青青点醒许仙", "青青救宣赞", "青青背宣赞", "小青复仇", "小青怒斥",
            "小青护法", "小青撮合", "小青婚礼", "小青担忧", "小青劝酒", "小青护法心",
            "白蛇报恩", "白素贞寻夫", "白蛇报恩银", "宿命姻缘", "白蛇前身", "素贞下凡",
            "素贞觅恩人", "金母赐锦囊", "素贞立誓", "白蛇报恩终", "许宣报恩",
        ],
        "rules": (
            "提取能够揭示文章核心意旨、人物特质、情感基调或作者议论抒情的句子。主题句子的实体原文必须一字不差引用原文完整原句，禁止删改字词、禁止用自己的话总结、禁止加标点断句。并在概括词列给出该主题的概括名（如'报恩''人妖恋'）。"
        ),
        "examples": [
            ("愿随侍左右。", [
                {"entity": "愿随侍左右。", "category": "motif", "explanation": "青青侍奉"},
            ]),
            ("小青复仇与反抗。", [
                {"entity": "小青复仇与反抗。", "category": "motif", "explanation": "小青复仇"},
            ]),
            ("功完方许度仙街。", [
                {"entity": "功完方许度仙街。", "category": "motif", "explanation": "白蛇报恩终"},
            ]),
        ],
    },
}


# ============================================================================
#  公开辅助函数（供 annotate_passage / compare_and_annotate 直接 import 调用）
# ============================================================================

def normalize_category(raw: str) -> str:
    """把任意描述（"人物"/"人名"/"person"）规范化为标准 category key。"""
    if not raw:
        return "custom"
    key = str(raw).strip()
    return _CATEGORY_REVERSE_MAP.get(key) or _CATEGORY_REVERSE_MAP.get(key.lower()) or "custom"


def get_category_label(category: str) -> str:
    """返回某类别的中文标签。"""
    kb = _ENTITY_KB.get(category or "")
    return kb["label"] if kb else (category or "自定义")


def list_categories() -> list[dict]:
    """返回全部标注类别的元信息列表（label/规则/示例）。"""
    return [
        {
            "key": cat,
            "label": kb.get("label", cat),
            "rules": kb.get("rules", ""),
            "examples": kb.get("examples", []),
            "knownCount": len(kb.get("known", [])),
            "color": DEFAULT_CATEGORY_COLORS.get(cat, "#B8B878"),
        }
        for cat, kb in _ENTITY_KB.items()
    ]


def get_known_entities(category: str = "", limit: int = 200) -> list[str]:
    """查询知识库中已知的实体词。category 空=全部。"""
    if category:
        return list(_ENTITY_KB.get(category, {}).get("known", []))[:limit]
    result: list[str] = []
    for kb in _ENTITY_KB.values():
        result.extend(kb.get("known", []))
    return result[:limit]


def lookup_entity(name: str) -> list[dict]:
    """查一个实体名在知识库中的所有命中（精确+包含）。"""
    if not name:
        return []
    hits = []
    n = name.strip()
    for cat, kb in _ENTITY_KB.items():
        for ent in kb.get("known", []):
            if n == ent or n in ent or ent in n:
                hits.append({"category": cat, "label": kb.get("label", ""), "entity": ent})
    return hits


# ============================================================================
#  MCP 工具封装: annotation_knowledge_base
#  查询模式: list_categories / lookup_entity / get_known_entities / category_rules
# ============================================================================

TOOL_DEF = {
    "name": "annotation_knowledge_base",
    "description": (
        "标注领域知识库查询。提供白蛇传标准的 5 类标注规范（人物/地点/事件/术语/母题），"
        "含类别规则、已知实体词表、few-shot 示例。"
        "当你不确定某个实体应该分到 person 还是 term、某条标注口径是什么时调用此工具，"
        "保证跨章节/跨会话的标注一致。"
        "典型问题：'XX 属于哪类？''母题的标注规则是什么？''白蛇传已知有哪些地点？'"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query_type": {
                "type": "string",
                "enum": [
                    "list_categories",
                    "category_rules",
                    "lookup_entity",
                    "get_known_entities",
                    "normalize_category",
                ],
                "description": (
                    "查询类型: "
                    "list_categories=所有类别元信息; "
                    "category_rules=取某个类别 category_key 的规则+示例; "
                    "lookup_entity=查 entity_name 在知识库中的命中（类别+同义词）; "
                    "get_known_entities=取 category 的已知实体词表; "
                    "normalize_category=将 raw_category 中文字符串（如'人物名称'）映射为标准 key。"
                ),
            },
            "category": {
                "type": "string",
                "description": "类别 key: person/location/event/term/motif/custom。category_rules / get_known_entities 需要。",
            },
            "entity_name": {
                "type": "string",
                "description": "实体名，lookup_entity 时必填。",
            },
            "raw_category": {
                "type": "string",
                "description": "normalize_category 模式下的原始中文字符串（如'人物''专有术语'）。",
            },
            "limit": {
                "type": "integer",
                "description": "get_known_entities 返回数量上限（默认 100）。",
            },
        },
        "required": ["query_type"],
    },
}


async def handler(query_type: str,
                  category: str = "",
                  entity_name: str = "",
                  raw_category: str = "",
                  limit: int = 100) -> dict:
    """MCP handler: 知识库查询。"""
    qt = (query_type or "").strip()
    if qt == "list_categories":
        return {"type": "list_categories", "categories": list_categories()}
    if qt == "category_rules":
        cat = normalize_category(category) if category else ""
        if not cat or cat not in _ENTITY_KB:
            return {"error": f"未知类别 category={category}；可用: {sorted(_ENTITY_KB.keys())}"}
        kb = _ENTITY_KB[cat]
        return {
            "type": "category_rules",
            "category": cat,
            "label": kb.get("label", ""),
            "color": DEFAULT_CATEGORY_COLORS.get(cat, "#B8B878"),
            "rules": kb.get("rules", ""),
            "examples": kb.get("examples", []),
            "knownSample": kb.get("known", [])[:30],
            "knownTotal": len(kb.get("known", [])),
        }
    if qt == "lookup_entity":
        if not entity_name:
            return {"error": "lookup_entity 需传入 entity_name"}
        hits = lookup_entity(entity_name)
        return {
            "type": "lookup_entity",
            "query": entity_name,
            "hits": hits,
            "hitCount": len(hits),
        }
    if qt == "get_known_entities":
        cat = normalize_category(category) if category else ""
        entities = get_known_entities(cat, limit=limit or 100)
        return {
            "type": "get_known_entities",
            "category": cat if cat else "(all)",
            "entities": entities,
            "count": len(entities),
        }
    if qt == "normalize_category":
        return {
            "type": "normalize_category",
            "raw": raw_category,
            "normalized": normalize_category(raw_category),
            "hint": "返回 standard key（person/location/event/term/motif/custom）。写标注/存数据库时用 normalized 值。",
        }
    return {"error": f"未知 query_type={qt!r}"}

