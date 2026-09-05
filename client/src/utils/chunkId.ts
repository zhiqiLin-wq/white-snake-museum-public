/** U17: chunk_id 解析工具 —— TypeScript 前端版本 */

export interface ParsedChunkId {
  sourcePrefix: 'primary' | 'research'
  sourceType: 'primary_literature' | 'research_literature'
  genre: string
  chapterNumber: string  // 字符串，不保证是数字
  sequence: number
}

export function parseChunkId(chunkId: string): ParsedChunkId {
  const parts = chunkId.split('_')
  if (parts.length < 4) {
    throw new Error(`Invalid chunk_id format: ${chunkId}`)
  }
  const sourcePrefix = parts[0] as 'primary' | 'research'
  const seqStr = parts[parts.length - 1]
  const chapterNumber = parts[parts.length - 2]
  const genre = parts.slice(1, -2).join('_')
  const sequence = parseInt(seqStr, 10)
  if (isNaN(sequence)) {
    throw new Error(`Invalid sequence in chunk_id: ${chunkId}`)
  }
  return {
    sourcePrefix,
    sourceType: sourcePrefix === 'primary' ? 'primary_literature' : 'research_literature',
    genre,
    chapterNumber,
    sequence,
  }
}
