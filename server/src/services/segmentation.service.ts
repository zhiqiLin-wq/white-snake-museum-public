import fs from 'node:fs';
import path from 'node:path';
import nodejieba from 'nodejieba';
import { STOP_WORDS } from '../utils/stopwords.js';
import type { Keyword, ExcelRecord, LocationData } from '../types/index.js';
import { parseExcelFile } from './excel.service.js';
import { inferLocationName } from '../utils/location-names.js';

/**
 * 对文本进行 jieba 分词，过滤停用词后提取 Top-10 关键词
 */
export function segmentAndRank(text: string): Keyword[] {
  const words = nodejieba.cut(text);
  const filtered = words.filter((w: string) => {
    if (w.length < 2) return false;
    if (/^\d+$/.test(w)) return false;
    if (STOP_WORDS.has(w)) return false;
    return true;
  });

  const freq: Record<string, number> = {};
  for (const w of filtered) {
    freq[w] = (freq[w] || 0) + 1;
  }

  return Object.entries(freq)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([word, count]) => ({ word, weight: count }));
}

/**
 * 扫描 excel_data 目录，解析所有 Excel 文件，
 * 聚合同名地点，计算关键词，返回 LocationData[]
 */
export function buildLocationData(excelDir: string): LocationData[] {
  const locationMap = new Map<string, { records: ExcelRecord[] }>();

  const files = fs.readdirSync(excelDir).filter(
    f => f.endsWith('.xlsx') || f.endsWith('.xls')
  );

  for (const file of files) {
    const records = parseExcelFile(path.join(excelDir, file));
    if (records.length === 0) continue;

    let name = inferLocationName(file);
    if (!name) name = path.basename(file, path.extname(file));

    const existing = locationMap.get(name);
    if (existing) {
      existing.records.push(...records);
    } else {
      locationMap.set(name, { records });
    }
  }

  return Array.from(locationMap.entries()).map(([name, { records }]) => {
    const totalMatch = records.reduce((sum, r) => sum + (r.匹配词数 || 1), 0);
    const totalWords = records.reduce((sum, r) => sum + (r.文本字数 || 0), 0);

    const fullText = records.map(r => r.摘要 || '').join('。');
    const keywords = segmentAndRank(fullText);

    return { name, records, totalMatch, totalWords, keywords };
  });
}
