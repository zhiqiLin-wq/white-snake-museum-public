import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import type { FastifyInstance } from 'fastify';
import { buildLocationData } from '../services/segmentation.service.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const EXCEL_DIR = path.resolve(__dirname, '../../../excel_data');

export async function locationRoutes(app: FastifyInstance): Promise<void> {
  app.get('/api/locations', async (_req, reply) => {

    if (!fs.existsSync(EXCEL_DIR)) {
      return reply.status(500).send({ error: `目录不存在: ${EXCEL_DIR}` });
    }

    try {
      const result = buildLocationData(EXCEL_DIR);
      return reply.send(result);
    } catch (err: any) {
      return reply.status(500).send({ error: err.message });
    }
  });
}
