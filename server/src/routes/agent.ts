/**
 * Node SSE 代理路由 — 透传 Python Agent 的流式对话到浏览器。
 *
 * 关键实现：
 * - 逐块透传（非缓冲），保持 SSE 的实时性
 * - 客户端断开 → 取消上游请求
 * - Python 服务不可用时返回 SSE 格式的错误事件
 * - B-100: /chat/resume 代理 (v8)
 */
import type { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';

const AGENT_URL = process.env.AGENT_URL || 'http://127.0.0.1:8000';

// B-102: 合法的 interrupt_id 值
const VALID_INTERRUPT_IDS = ['confirm_motifs', 'select_hypothesis', 'decide_next_action'];

export async function agentRoutes(app: FastifyInstance): Promise<void> {

  // ================================================================
  //  POST /api/agent/chat — SSE 流式对话
  // ================================================================
  app.post('/api/agent/chat', async (req: FastifyRequest, reply: FastifyReply) => {
    const body = req.body as {
      query: string;
      skill?: string;
      stream?: boolean;
      conversation_id?: string;
    };

    // 设置 SSE 响应头（含 CORS 头，因 bypass Fastify CORS 中间件）
    reply.raw.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'X-Accel-Buffering': 'no',     // 禁用 nginx 缓冲
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    });

    const ac = new AbortController();

    // 客户端断开 → 取消上游请求
    req.raw.on('close', () => {
      ac.abort();
    });

    try {
      const upstream = await fetch(`${AGENT_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(req.userId ? { 'x-user-id': req.userId } : {}),
        },
        body: JSON.stringify({ ...body, stream: true, user_id: req.userId || '' }),
        signal: ac.signal,
      });

      if (!upstream.ok || !upstream.body) {
        // Python 服务不可用时的 SSE 错误响应
        const errorMsg = JSON.stringify({
          type: 'error',
          data: {
            code: 'AGENT_UNAVAILABLE',
            message: 'AI 服务暂时不可用，请稍后重试',
          },
        });
        reply.raw.write(`event: error\ndata: ${errorMsg}\n\n`);
        reply.raw.write(`event: done\ndata: {"workflow":"none"}\n\n`);
        reply.raw.end();
        return;
      }

      // ★ 关键：逐块透传，不做缓冲
      const reader = upstream.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        reply.raw.write(chunk);
      }

      reply.raw.end();

    } catch (err: any) {
      if (err.name === 'AbortError') {
        // 客户端正常断开，不记录错误
        return;
      }
      app.log.error({ err }, 'SSE proxy error');
      if (!reply.raw.writableEnded) {
        reply.raw.write(
          `event: error\ndata: ${JSON.stringify({ message: '代理层错误' })}\n\n`
        );
        reply.raw.end();
      }
    }
  });

  // ================================================================
  //  B-100: POST /api/agent/chat/resume — 中断恢复代理 (v8)
  // ================================================================
  app.post('/api/agent/chat/resume', async (req: FastifyRequest, reply: FastifyReply) => {
    const body = req.body as {
      thread_id: string;
      interrupt_id: string;
      resume: {
        action: string;
        edited_motifs?: Array<Record<string, unknown>>;
        selected_hypothesis_ids?: string[];
        decision?: string;
      };
    };

    // B-102: 请求体验证
    if (!body.thread_id || typeof body.thread_id !== 'string' || body.thread_id.trim() === '') {
      return reply.status(400).send({ error: 'BAD_REQUEST', message: 'thread_id is required' });
    }
    if (!body.interrupt_id || !VALID_INTERRUPT_IDS.includes(body.interrupt_id)) {
      return reply.status(400).send({
        error: 'BAD_REQUEST',
        message: `interrupt_id must be one of: ${VALID_INTERRUPT_IDS.join(', ')}`,
      });
    }
    if (!body.resume || !body.resume.action) {
      return reply.status(400).send({ error: 'BAD_REQUEST', message: 'resume.action is required' });
    }

    // 设置 SSE 响应头
    reply.raw.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'X-Accel-Buffering': 'no',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    });

    // B-101: AbortController 支持 — 客户端断开时取消上游请求
    const ac = new AbortController();
    req.raw.on('close', () => {
      ac.abort();
    });

    try {
      const upstream = await fetch(`${AGENT_URL}/chat/resume`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(req.userId ? { 'x-user-id': req.userId } : {}),
        },
        body: JSON.stringify({ ...body, user_id: req.userId || '' }),
        signal: ac.signal,
      });

      if (!upstream.ok || !upstream.body) {
        const statusCode = upstream.status;
        let errorMsg: string;
        if (statusCode === 404) {
          errorMsg = JSON.stringify({
            type: 'error',
            data: { code: 'THREAD_NOT_FOUND', message: '会话未找到' },
          });
        } else if (statusCode === 410) {
          errorMsg = JSON.stringify({
            type: 'error',
            data: { code: 'SESSION_TIMEOUT', message: '会话已超时，请重新发起演化追踪' },
          });
        } else {
          errorMsg = JSON.stringify({
            type: 'error',
            data: { code: 'RESUME_FAILED', message: '恢复失败，请重试' },
          });
        }
        reply.raw.write(`event: error\ndata: ${errorMsg}\n\n`);
        reply.raw.write(`event: done\ndata: {"workflow":"none"}\n\n`);
        reply.raw.end();
        return;
      }

      // 逐块透传 SSE 流
      const reader = upstream.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        reply.raw.write(chunk);
      }

      reply.raw.end();

    } catch (err: any) {
      if (err.name === 'AbortError') {
        return;  // B-101: 客户端正常断开
      }
      app.log.error({ err }, 'Resume SSE proxy error');
      if (!reply.raw.writableEnded) {
        reply.raw.write(
          `event: error\ndata: ${JSON.stringify({ message: '代理层错误' })}\n\n`
        );
        reply.raw.end();
      }
    }
  });

  // ================================================================
  //  POST /api/agent/chat/sync — 非流式对话
  // ================================================================
  app.post('/api/agent/chat/sync', async (req, reply) => {
    try {
      const body = req.body as Record<string, unknown>
      const resp = await fetch(`${AGENT_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(req.userId ? { 'x-user-id': req.userId } : {}),
        },
        body: JSON.stringify({ ...body, stream: false, user_id: req.userId || '' }),
        signal: AbortSignal.timeout(120_000),   // 2 分钟超时
      });

      if (!resp.ok) {
        return reply.status(502).send({
          error: 'AGENT_UNAVAILABLE',
          message: 'AI 服务暂时不可用',
        });
      }

      return reply.send(await resp.json());
    } catch (err) {
      return reply.status(502).send({
        error: 'AGENT_UNREACHABLE',
        message: '无法连接到 AI 服务',
      });
    }
  });

  // ================================================================
  //  GET /api/agent/chat/evolution-state — 演化分析历史状态恢复代理
  //  从 Python 端 LangGraph checkpointer (SQLite) 按 threadId 读取完整 state
  // ================================================================
  app.get('/api/agent/chat/evolution-state', async (req, reply) => {
    const query = req.query as { thread_id?: string };
    if (!query.thread_id || query.thread_id.trim() === '') {
      return reply.status(400).send({ error: 'BAD_REQUEST', message: 'thread_id is required' });
    }
    try {
      const resp = await fetch(
        `${AGENT_URL}/chat/evolution-state?thread_id=${encodeURIComponent(query.thread_id)}`,
        {
          headers: {
            ...(req.userId ? { 'x-user-id': req.userId } : {}),
          },
          signal: AbortSignal.timeout(10_000),
        }
      );
      if (!resp.ok) {
        const payload = await resp.json().catch(() => ({ error: 'UPSTREAM_ERROR' }));
        return reply.status(resp.status).send(payload);
      }
      return reply.send(await resp.json());
    } catch (err) {
      app.log.error({ err }, 'evolution-state proxy error');
      return reply.status(502).send({ error: 'AGENT_UNREACHABLE', message: '无法连接到 AI 服务' });
    }
  });

  // ================================================================
  //  GET /api/agent/skills — 技能列表
  // ================================================================
  app.get('/api/agent/skills', async (req, reply) => {
    try {
      const resp = await fetch(`${AGENT_URL}/skills?user_id=${encodeURIComponent(req.userId || '')}`);
      return reply.send(await resp.json());
    } catch {
      return reply.status(502).send({ error: 'AGENT_UNREACHABLE' });
    }
  });

  // ================================================================
  //  GET /api/agent/health — 健康检查
  // ================================================================
  app.get('/api/agent/health', async (req, reply) => {
    try {
      const resp = await fetch(`${AGENT_URL}/health?user_id=${encodeURIComponent(req.userId || '')}`, {
        signal: AbortSignal.timeout(5000),
      });
      return reply.send(await resp.json());
    } catch {
      return reply.status(502).send({
        status: 'degraded',
        agent: 'unreachable',
      });
    }
  });

  // ================================================================
  //  v19: GET /api/agent/reports — 报告文件库列表
  // ================================================================
  app.get('/api/agent/reports', async (req, reply) => {
    try {
      const resp = await fetch(`${AGENT_URL}/reports`, {
        headers: {
          ...(req.userId ? { 'x-user-id': req.userId } : {}),
        },
        signal: AbortSignal.timeout(10_000),
      });
      return reply.send(await resp.json());
    } catch (err) {
      app.log.error({ err }, 'reports list proxy error');
      return reply.status(502).send({ error: 'AGENT_UNREACHABLE', message: '无法连接到 AI 服务' });
    }
  });

  // ================================================================
  //  v19: GET /api/agent/reports/:reportId — 读取报告全文（平台内预览）
  // ================================================================
  app.get('/api/agent/reports/:reportId', async (req, reply) => {
    const { reportId } = req.params as { reportId: string };
    if (!reportId || !/^[A-Za-z0-9_-]+$/.test(reportId)) {
      return reply.status(400).send({ error: 'BAD_REQUEST', message: 'reportId is required' });
    }
    try {
      const resp = await fetch(`${AGENT_URL}/reports/${encodeURIComponent(reportId)}`, {
        headers: {
          ...(req.userId ? { 'x-user-id': req.userId } : {}),
        },
        signal: AbortSignal.timeout(15_000),
      });
      if (!resp.ok) {
        const payload = await resp.json().catch(() => ({ error: 'UPSTREAM_ERROR' }));
        return reply.status(resp.status).send(payload);
      }
      return reply.send(await resp.json());
    } catch (err) {
      app.log.error({ err }, 'report get proxy error');
      return reply.status(502).send({ error: 'AGENT_UNREACHABLE', message: '无法连接到 AI 服务' });
    }
  });

  // ================================================================
  //  v19: DELETE /api/agent/reports/:reportId — 删除报告文件
  // ================================================================
  app.delete('/api/agent/reports/:reportId', async (req, reply) => {
    const { reportId } = req.params as { reportId: string };
    if (!reportId || !/^[A-Za-z0-9_-]+$/.test(reportId)) {
      return reply.status(400).send({ error: 'BAD_REQUEST', message: 'reportId is required' });
    }
    try {
      const resp = await fetch(`${AGENT_URL}/reports/${encodeURIComponent(reportId)}`, {
        method: 'DELETE',
        headers: {
          ...(req.userId ? { 'x-user-id': req.userId } : {}),
        },
        signal: AbortSignal.timeout(10_000),
      });
      if (!resp.ok) {
        const payload = await resp.json().catch(() => ({ error: 'UPSTREAM_ERROR' }));
        return reply.status(resp.status).send(payload);
      }
      return reply.send(await resp.json());
    } catch (err) {
      app.log.error({ err }, 'report delete proxy error');
      return reply.status(502).send({ error: 'AGENT_UNREACHABLE', message: '无法连接到 AI 服务' });
    }
  });

  // ================================================================
  //  POST /api/agent/rebuild-index — 重建向量索引
  // ================================================================
  app.post('/api/agent/rebuild-index', async (req, reply) => {
    try {
      const body = req.body as Record<string, unknown>
      const resp = await fetch(`${AGENT_URL}/rebuild-index`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(req.userId ? { 'x-user-id': req.userId } : {}),
        },
        body: JSON.stringify({ ...body, user_id: req.userId || '' }),
        signal: AbortSignal.timeout(300_000),  // 重建索引可能较慢
      });
      return reply.send(await resp.json());
    } catch {
      return reply.status(502).send({ error: 'AGENT_UNREACHABLE' });
    }
  });
}
