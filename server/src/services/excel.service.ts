import fs from 'node:fs';
import path from 'node:path';
import XLSX from 'xlsx';
import type { ExcelRecord } from '../types/index.js';

/**
 * 解析单个 Excel 分词结果文件
 *
 * 支持两种格式：
 *   A. 金山寺格式：header 含「地点名」「序号」→ 直接从"文本内容"列提取
 *   B. 通用格式：header 含 A/B/C/D 或 切词/匹配词/字数/文本内容
 */
export function parseExcelFile(filePath: string): ExcelRecord[] {
  const workbook = XLSX.readFile(filePath);
  const sheetName = workbook.SheetNames[0];
  const sheet = workbook.Sheets[sheetName];
  const rows: unknown[][] = XLSX.utils.sheet_to_json(sheet, { header: 1, defval: '' });

  if (!rows || rows.length < 2) return [];

  const header: string[] = (rows[0] as unknown[]).map(cell => String(cell ?? '').trim());

  // ----- 金山寺格式 -----
  const isJinshanFormat = header.includes('地点名') && header.includes('序号');
  if (isJinshanFormat) {
    const textColIndex = header.findIndex(h => h.includes('文本内容'));
    if (textColIndex === -1) return [];

    const records: ExcelRecord[] = [];
    for (let i = 1; i < rows.length; i++) {
      const row = rows[i] as unknown[];
      if (!row || row.length <= textColIndex) continue;
      const content = row[textColIndex] ? String(row[textColIndex]).trim() : '';
      if (content.length === 0) continue;
      records.push({
        切词数: 0,
        匹配词数: 1,
        文本字数: content.length,
        摘要: content.substring(0, 300),
      });
    }
    return records;
  }

  // ----- 通用格式 -----
  const colToken = header.findIndex(h => h.includes('切词') || h === 'A');
  const colMatch = header.findIndex(h => h.includes('匹配词') || h === 'B');
  const colWords = header.findIndex(h => h.includes('字数') || h === 'C');
  const colText  = header.findIndex(h => h.includes('文本内容') || h === 'D');

  if (colToken === -1 || colMatch === -1 || colText === -1) return [];

  const records: ExcelRecord[] = [];
  for (let i = 1; i < rows.length; i++) {
    const row = rows[i] as unknown[];
    if (!row || row.length < 4) continue;

    const tokens  = parseInt(String(row[colToken]))  || 0;
    const matches = parseInt(String(row[colMatch]))  || 0;
    const words   = parseInt(String(row[colWords]))  || 0;
    const content = row[colText] ? String(row[colText]).trim() : '';

    if (content.length === 0 && tokens === 0 && matches === 0) continue;

    records.push({
      切词数: tokens,
      匹配词数: matches,
      文本字数: words,
      摘要: content.substring(0, 300),
    });
  }
  return records;
}
