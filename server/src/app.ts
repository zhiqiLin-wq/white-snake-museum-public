import Fastify from 'fastify';
import cors from '@fastify/cors';
import fastifyStatic from '@fastify/static';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { literatureRoutes } from './routes/literature.js';
import { locationRoutes } from './routes/locations.js';
import { agentRoutes } from './routes/agent.js';
import { authRoutes } from './routes/auth.js';
import { annotationRoutes } from './routes/annotations.js';
import { conversationRoutes } from './routes/conversations.js';
import { authMiddleware } from './middleware/auth.js';
import { initAuthDB, startSessionCleanup } from './services/auth.service.js';
import { migrateAllUserDBs } from './services/user-data.service.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = 3000;

async function main() {
  const app = Fastify({ logger: { level: 'info' } });

  // CORS（开发时允许 Vite dev server 跨域）
  await app.register(cors, { origin: true });

  // 静态文件：优先从 client/dist（构建产物），回退到 public（兼容旧版）
  const clientDist = path.resolve(__dirname, '../../client/dist');
  const publicDir  = path.resolve(__dirname, '../../public');

  const staticRoots: string[] = [];
  if (fs.existsSync(clientDist)) staticRoots.push(clientDist);
  if (fs.existsSync(publicDir)) staticRoots.push(publicDir);

  if (staticRoots.length > 0) {
    await app.register(fastifyStatic, {
      root: staticRoots,
      prefix: '/',
    });
  }

  initAuthDB();
  migrateAllUserDBs();   // 启动时扫描所有用户数据库，执行 schema 升级迁移
  startSessionCleanup();

  await app.register(authRoutes);

  app.addHook('preHandler', authMiddleware);

  await app.register(annotationRoutes);
  await app.register(conversationRoutes);

  await app.register(literatureRoutes);
  await app.register(locationRoutes);
  await app.register(agentRoutes);

  try {
    await app.listen({ port: PORT, host: '0.0.0.0' });
    console.log(`后端已启动: http://localhost:${PORT}`);
    console.log(`数据目录: ${path.resolve(__dirname, '../../excel_data')}`);
  } catch (err) {
    app.log.error(err);
    process.exit(1);
  }
}

main();
