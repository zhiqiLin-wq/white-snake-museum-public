/**
 * 提取 文本景观建构研究文献.docx → .txt
 * 使用 mammoth (已安装的 npm 包)
 */
const mammoth = require('mammoth');
const fs = require('fs');
const path = require('path');

const DOCX = path.join(__dirname, '..', 'excel_data', '文本景观建构研究文献.docx');
const TXT  = path.join(__dirname, '..', 'excel_data', '文本景观建构研究文献.txt');

async function extract() {
  console.log('提取中...');
  const result = await mammoth.extractRawText({ path: DOCX });
  fs.writeFileSync(TXT, result.value, 'utf-8');
  console.log(`完成: ${result.value.length} 字符 → ${TXT}`);
}

extract().catch(e => { console.error('提取失败:', e.message); process.exit(1); });
