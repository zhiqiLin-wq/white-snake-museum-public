// ==================== 文献类型 ====================

export interface Chapter {
  number: string;          // 章节编号，如 "一"、"二" … 或 "全文"
  chapterNumber: number;   // 数值型章节编号，如 1, 2 … 供程序化查找
  title: string;           // 完整标题行
  content: string;         // 章节正文
}

// ==================== Excel 解析类型 ====================

export interface ExcelRecord {
  /** 分词数 */
  切词数: number;
  /** 匹配词数 */
  匹配词数: number;
  /** 文本字数 */
  文本字数: number;
  /** 文本摘要（最多300字） */
  摘要: string;
}

export interface Keyword {
  word: string;
  weight: number;
}

export interface LocationData {
  name: string;
  records: ExcelRecord[];
  totalMatch: number;
  totalWords: number;
  keywords: Keyword[];
}

// ==================== Fastify 类型扩展 ====================

declare module 'fastify' {
  interface FastifyRequest {
    userId?: string
  }
}
