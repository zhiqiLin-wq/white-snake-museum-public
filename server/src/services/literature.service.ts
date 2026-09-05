import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import mammoth from 'mammoth';
import type { Chapter } from '../types/index.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DATA_DIR = path.resolve(__dirname, '../../../excel_data');

const LITERATURE_TXT  = path.join(DATA_DIR, '白蛇传文献选集.txt');
const LITERATURE_DOCX = path.join(DATA_DIR, '白蛇传文献选集.docx');

const CHINESE_NUMBERS: Record<string, number> = {
  '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7,
};

function parseChapterNumber(raw: string): number {
  return CHINESE_NUMBERS[raw.trim()] || parseInt(raw, 10) || 0;
}

// ★ 新增：文本景观建构研究文献
const RESEARCH_LITERATURE_TXT  = path.join(DATA_DIR, '文本景观建构研究文献.txt');
const RESEARCH_LITERATURE_DOCX = path.join(DATA_DIR, '文本景观建构研究文献.docx');

/**
 * 解析文献正文为章节数组（严格七章版）
 *
 * 匹配规则：以「一」~「七」编号开头的行视为章节标题。
 * 如果一个都没匹配到，整篇作为单章返回。
 */
export function parseLiterature(text: string): Chapter[] {
  const regex = /^\s*(一|二|三|四|五|六|七)\s*[、，,.\s]\s*(.+)$/gm;

  interface TitleMeta { index: number; fullTitle: string; number: string; title: string }
  const titles: TitleMeta[] = [];

  let match: RegExpExecArray | null;
  while ((match = regex.exec(text)) !== null) {
    titles.push({
      index: match.index,
      fullTitle: match[0].trim(),
      number:  match[1].trim(),
      title:   match[2].trim(),
    });
  }

  if (titles.length === 0) {
    console.log('未匹配到任何章节，文本前200字符：', text.substring(0, 200));
    return [];
  }

  const chapters: Chapter[] = [];
  let lastIndex = 0;

  for (let i = 0; i < titles.length; i++) {
    const start = text.indexOf(titles[i].fullTitle, lastIndex);
    const end = i < titles.length - 1
      ? text.indexOf(titles[i + 1].fullTitle, start + titles[i].fullTitle.length)
      : text.length;

    chapters.push({
      number:  titles[i].number,
      chapterNumber: parseChapterNumber(titles[i].number),
      title:   titles[i].fullTitle,
      content: text.substring(start + titles[i].fullTitle.length, end).trim(),
    });
    lastIndex = end;
  }

  return chapters;
}

/**
 * 读取文献文件（优先 .txt，其次 .docx）
 */
export async function loadLiteratureText(): Promise<string> {
  if (fs.existsSync(LITERATURE_TXT)) {
    return fs.readFileSync(LITERATURE_TXT, 'utf8');
  }
  if (fs.existsSync(LITERATURE_DOCX)) {
    const result = await mammoth.extractRawText({ path: LITERATURE_DOCX });
    return result.value;
  }
  throw new Error('文献文件不存在（.txt 或 .docx）');
}

/**
 * ★ 新增：读取文本景观建构研究文献（优先 .txt，其次 .docx）
 */
export async function loadResearchLiteratureText(): Promise<string> {
  if (fs.existsSync(RESEARCH_LITERATURE_TXT)) {
    return fs.readFileSync(RESEARCH_LITERATURE_TXT, 'utf8');
  }
  if (fs.existsSync(RESEARCH_LITERATURE_DOCX)) {
    const result = await mammoth.extractRawText({ path: RESEARCH_LITERATURE_DOCX });
    return result.value;
  }
  throw new Error('研究文献文件不存在（.txt 或 .docx）');
}
