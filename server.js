/*
 * DEPRECATED: This file is the legacy Express server.
 *
 * The active server is server/src/app.ts (Fastify + SQLite + multi-user auth).
 * This file is kept ONLY for the /api/literature and /api/locations endpoints
 * as a read-only fallback. All annotation routes have been migrated to the
 * Fastify server. Do NOT add new routes here.
 *
 * Start the active server with: cd server && npm run dev
 */

const express = require('express');
const fs = require('fs');
const path = require('path');
const XLSX = require('xlsx');
const cors = require('cors');
const nodejieba = require('nodejieba');
const mammoth = require('mammoth');

const app = express();
const PORT = 3001;  // 避免与 Fastify 主服务器 (port 3000) 冲突

app.use(cors());
app.use(express.static('public'));

const EXCEL_DIR = path.join(__dirname, 'excel_data');
const LITERATURE_TXT = path.join(__dirname, 'excel_data', '白蛇传文献选集.txt');
const LITERATURE_DOCX = path.join(__dirname, 'excel_data', '白蛇传文献选集.docx');
const RESEARCH_TXT = path.join(__dirname, 'excel_data', '文本景观建构研究文献.txt');
const RESEARCH_DOCX = path.join(__dirname, 'excel_data', '文本景观建构研究文献.docx');

const STOP_WORDS = new Set([
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
    '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',
    '没有', '看', '好', '自己', '这', '他', '她', '它', '们', '那', '些',
    '所', '为', '所以', '因为', '但是', '然而', '可以', '这个', '那个',
    '已经', '还是', '只是', '什么', '怎么', '如何', '为什么', '如果',
    '虽然', '而且', '或', '但', '与', '及', '其', '之', '从', '以', '等',
    '又', '将', '被', '把', '向', '对', '于', '啊', '呢', '吧', '吗',
    '哦', '嗯', '哈', '呀', '哇', '哎', '哟', '啦', '噢', '过',
    '得', '地',
]);

// ============= 文献解析（严格七章版） =============
function parseLiterature(text) {
    const chapters = [];
    // 只匹配“一”到“七”开头，后跟顿号/逗号/空格/英文点，再跟标题内容
    const regex = /^\s*(一|二|三|四|五|六|七)\s*[、，,.\s]\s*(.+)$/gm;
    let match;
    let lastIndex = 0;
    const titles = [];

    while ((match = regex.exec(text)) !== null) {
        const fullTitle = match[0].trim();
        const numberPart = match[1].trim();
        const titlePart = match[2].trim();
        titles.push({
            index: match.index,
            fullTitle: fullTitle,
            number: numberPart,
            title: titlePart
        });
    }

    if (titles.length === 0) {
        // 如果一个都没匹配到，打印文本前200字符方便诊断
        console.log('未匹配到任何章节，文本前200字符：', text.substring(0, 200));
    }

    for (let i = 0; i < titles.length; i++) {
        const start = text.indexOf(titles[i].fullTitle, lastIndex);
        const end = i < titles.length - 1
            ? text.indexOf(titles[i + 1].fullTitle, start + titles[i].fullTitle.length)
            : text.length;
        const content = text.substring(start + titles[i].fullTitle.length, end).trim();
        const chineseNumMap = { '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7 };
        chapters.push({
            number: titles[i].number,
            chapterNumber: chineseNumMap[titles[i].number] || parseInt(titles[i].number, 10) || 0,
            title: titles[i].fullTitle,
            content: content
        });
        lastIndex = end;
    }

    return chapters;
}

app.get('/api/literature/research', async (req, res) => {
    try {
        let text = '';
        if (fs.existsSync(RESEARCH_TXT)) {
            text = fs.readFileSync(RESEARCH_TXT, 'utf8');
        } else if (fs.existsSync(RESEARCH_DOCX)) {
            const result = await mammoth.extractRawText({ path: RESEARCH_DOCX });
            text = result.value;
        } else {
            return res.status(404).json({ error: '研究文献文件不存在' });
        }
        // 研究文献不分章节，按自然段落返回
        const paragraphs = text.split(/\n\n+/).filter(p => p.trim().length > 0);
        res.json([{
            number: '研究',
            title: '文本景观建构研究文献',
            content: text,
            paragraphs: paragraphs.map((p, i) => ({ index: i, text: p.trim() })),
        }]);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.get('/api/literature', async (req, res) => {
    try {
        let text = '';
        if (fs.existsSync(LITERATURE_TXT)) {
            text = fs.readFileSync(LITERATURE_TXT, 'utf8');
        } else if (fs.existsSync(LITERATURE_DOCX)) {
            const result = await mammoth.extractRawText({ path: LITERATURE_DOCX });
            text = result.value;
        } else {
            return res.status(404).json({ error: '文献文件不存在（.txt 或 .docx）' });
        }
        const chapters = parseLiterature(text);
        if (chapters.length === 0) {
            return res.json([{ number: '全文', title: '白蛇传文献选集', content: text }]);
        }
        res.json(chapters);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// ============= Excel 解析 =============
function inferLocationName(filename) {
    const name = filename.replace(/分词结果|\.xlsx|\.xls|的结果数据/gi, '').trim();
    if (name.includes('西子湖')) return '西子湖';
    if (name.includes('西湖')) return '西湖';
    if (name.includes('峨眉')) return '峨眉山';
    if (name.includes('青城')) return '青城山';
    if (name.includes('金山寺')) return '金山寺';
    if (name.includes('雷峰塔')) return '雷峰塔';
    if (name.includes('承天寺')) return '承天寺';
    if (name.includes('卧佛寺')) return '卧佛寺';
    if (name.includes('望江楼')) return '望江楼';
    if (name.includes('龙虎山')) return '龙虎山';
    if (name.includes('灵隐寺')) return '灵隐寺';
    if (name.includes('断桥')) return '断桥';
    return name;
}

function parseExcelFile(filePath) {
    const workbook = XLSX.readFile(filePath);
    const sheetName = workbook.SheetNames[0];
    const sheet = workbook.Sheets[sheetName];
    const rows = XLSX.utils.sheet_to_json(sheet, { header: 1, defval: "" });
    if (!rows || rows.length < 2) return [];

    const header = rows[0].map(cell => String(cell || '').trim());
    const isJinshanFormat = header.includes('地点名') && header.includes('序号');

    if (isJinshanFormat) {
        const textColIndex = header.findIndex(h => h.includes('文本内容'));
        if (textColIndex === -1) return [];
        const records = [];
        for (let i = 1; i < rows.length; i++) {
            const row = rows[i];
            if (!row || row.length <= textColIndex) continue;
            let content = row[textColIndex] ? String(row[textColIndex]).trim() : '';
            if (content.length === 0) continue;
            records.push({ 切词数: 0, 匹配词数: 1, 文本字数: content.length, 摘要: content.substring(0, 300) });
        }
        return records;
    }

    const idxTokens = header.findIndex(h => h.includes('切词') || h === 'A');
    const idxMatch = header.findIndex(h => h.includes('匹配词') || h === 'B');
    const idxWordCount = header.findIndex(h => h.includes('字数') || h === 'C');
    const idxContent = header.findIndex(h => h.includes('文本内容') || h === 'D');

    if (idxTokens === -1 || idxMatch === -1 || idxContent === -1) return [];

    const records = [];
    for (let i = 1; i < rows.length; i++) {
        const row = rows[i];
        if (!row || row.length < 4) continue;
        const tokens = parseInt(row[idxTokens]) || 0;
        const match = parseInt(row[idxMatch]) || 0;
        const wordCnt = parseInt(row[idxWordCount]) || 0;
        let content = row[idxContent] ? String(row[idxContent]).trim() : '';
        if (content.length === 0 && tokens === 0 && match === 0) continue;
        records.push({ 切词数: tokens, 匹配词数: match, 文本字数: wordCnt, 摘要: content.substring(0, 300) });
    }
    return records;
}

app.get('/api/locations', (req, res) => {
    if (!fs.existsSync(EXCEL_DIR)) {
        return res.status(500).json({ error: `目录不存在: ${EXCEL_DIR}` });
    }
    const files = fs.readdirSync(EXCEL_DIR).filter(f => f.endsWith('.xlsx') || f.endsWith('.xls'));
    const locationMap = new Map();

    for (const file of files) {
        const filePath = path.join(EXCEL_DIR, file);
        const records = parseExcelFile(filePath);
        if (records.length === 0) continue;
        let locationName = inferLocationName(file);
        if (!locationName) locationName = path.basename(file, path.extname(file));

        if (locationMap.has(locationName)) {
            locationMap.get(locationName).records.push(...records);
        } else {
            locationMap.set(locationName, { name: locationName, records });
        }
    }

    const result = Array.from(locationMap.values()).map(loc => {
        const totalMatch = loc.records.reduce((sum, r) => sum + (r.匹配词数 || 1), 0);
        const totalWords = loc.records.reduce((sum, r) => sum + (r.文本字数 || 0), 0);

        const fullText = loc.records.map(r => r.摘要 || '').join('。');
        const words = nodejieba.cut(fullText);
        const filtered = words.filter(w => {
            if (w.length < 2) return false;
            if (/^\d+$/.test(w)) return false;
            if (STOP_WORDS.has(w)) return false;
            return true;
        });

        const freq = {};
        filtered.forEach(w => { freq[w] = (freq[w] || 0) + 1; });

        const keywords = Object.entries(freq)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 10)
            .map(([word, count]) => ({ word, weight: count }));

        return {
            name: loc.name,
            records: loc.records,
            totalMatch,
            totalWords,
            keywords
        };
    });

    res.json(result);
});

// ============= 标注端点已迁移到 server/src/routes/annotations.ts =============
// 所有 /api/annotations/* 路由由 Fastify + SQLite 服务器处理。
// 此文件仅保留 /api/literature 和 /api/locations 作为只读公共服务。

app.listen(PORT, () => {
    console.log(`[Legacy Express] Literature-only server: http://localhost:${PORT} (DEPRECATED - use Fastify on port 3000)`);
    console.log(`[Legacy Express] Excel: ${EXCEL_DIR}`);
    console.log(`[Legacy Express] WARNING: This is NOT the main server.`);
    console.log(`[Legacy Express] Use 'cd server && npm run dev' for the full Fastify server.`);
});