import type { FastifyInstance } from 'fastify';
import { loadLiteratureText, parseLiterature, loadResearchLiteratureText } from '../services/literature.service.js';

export async function literatureRoutes(app: FastifyInstance): Promise<void> {
  app.get('/api/literature', async (_req, reply) => {
    try {
      const text = await loadLiteratureText();
      const chapters = parseLiterature(text);

      if (chapters.length === 0) {
        return reply.send([{
          number: '全文',
          title: '白蛇传文献选集',
          content: text,
        }]);
      }

      return reply.send(chapters);
    } catch (err: any) {
      if (err.message?.includes('文件不存在')) {
        return reply.status(404).send({ error: err.message });
      }
      return reply.status(500).send({ error: err.message });
    }
  });

  // ★ 全文搜索 API
  app.get('/api/literature/search', async (req, reply) => {
    const q = (req.query as Record<string, string>).q || ''
    if (q.trim().length < 2) {
      return reply.status(400).send({ error: "查询参数 'q' 不能少于 2 个字符" })
    }

    try {
      const text = await loadLiteratureText()
      const chapters = parseLiterature(text)

      interface SearchResultItem {
        rank: number
        chapterNumber: string
        chapterTitle: string
        paragraphIndex: number
        excerpt: string
        highlightRanges: [number, number][]
        paragraphMatchPositions: [number, number][]
        relevanceScore: number
      }

      const results: SearchResultItem[] = []
      const query = q.trim()

      for (const ch of chapters) {
        // Normalize line endings before splitting, matching frontend splitParagraphs behavior
        const normalizedContent = ch.content.replace(/\r\n|\r/g, '\n')
        const paragraphs = normalizedContent.split(/\n\n/).map(p => p.trim()).filter(p => p.length > 0)
        for (let pi = 0; pi < paragraphs.length; pi++) {
          const paraText = paragraphs[pi]
          const lowerText = paraText.toLowerCase()
          const lowerQuery = query.toLowerCase()
          const matchPositions: [number, number][] = []

          let searchFrom = 0
          while (searchFrom < lowerText.length) {
            const idx = lowerText.indexOf(lowerQuery, searchFrom)
            if (idx === -1) break
            matchPositions.push([idx, idx + query.length])
            searchFrom = idx + 1
          }

          if (matchPositions.length > 0) {
            const firstMatch = matchPositions[0]
            const excerptStart = Math.max(0, firstMatch[0] - 20)
            const excerptEnd = Math.min(paraText.length, firstMatch[1] + 40)
            const excerpt = paraText.slice(excerptStart, excerptEnd)
            const highlightRanges: [number, number][] = matchPositions.map(([s, e]) => [s - excerptStart, e - excerptStart])
            const relevanceScore = matchPositions.length / Math.pow(paraText.length, 0.3)

            results.push({
              rank: 0,
              chapterNumber: String(ch.chapterNumber),
              chapterTitle: ch.title,
              paragraphIndex: pi,
              excerpt,
              highlightRanges,
              paragraphMatchPositions: matchPositions,
              relevanceScore,
            })
          }
        }
      }

      // Sort by relevance score descending
      results.sort((a, b) => b.relevanceScore - a.relevanceScore)
      // Assign ranks
      for (let i = 0; i < results.length; i++) {
        results[i].rank = i + 1
      }

      return reply.send({
        query,
        totalMatches: results.reduce((sum, r) => sum + r.paragraphMatchPositions.length, 0),
        results,
        sortOptions: ['相关度', '按朝代', '按章节顺序'],
      })
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : '搜索失败'
      return reply.status(500).send({ error: message })
    }
  })

  // ★ 新增：文本景观建构研究文献 API
  app.get('/api/research-literature', async (_req, reply) => {
    try {
      const text = await loadResearchLiteratureText();
      return reply.send({
        title: '文本景观建构研究文献',
        content: text,
        length: text.length,
      });
    } catch (err: any) {
      if (err.message?.includes('文件不存在')) {
        return reply.status(404).send({ error: err.message });
      }
      return reply.status(500).send({ error: err.message });
    }
  });
}
