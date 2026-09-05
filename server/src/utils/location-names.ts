/**
 * 从文件名推断地点名称
 */
const LOCATION_KEYWORDS: Readonly<Record<string, string>> = {
  '西子湖': '西子湖',
  '西湖':   '西湖',
  '峨眉':   '峨眉山',
  '青城':   '青城山',
  '金山寺': '金山寺',
  '雷峰塔': '雷峰塔',
  '承天寺': '承天寺',
  '卧佛寺': '卧佛寺',
  '望江楼': '望江楼',
  '龙虎山': '龙虎山',
  '灵隐寺': '灵隐寺',
  '断桥':   '断桥',
};

export function inferLocationName(filename: string): string {
  const clean = filename.replace(/分词结果|\.xlsx|\.xls|的结果数据/gi, '').trim();
  for (const [key, name] of Object.entries(LOCATION_KEYWORDS)) {
    if (clean.includes(key)) return name;
  }
  return clean;
}
