import { createReadStream, existsSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, normalize, resolve, sep } from 'node:path';
import { once } from 'node:events';

const TYPES = new Map([
  ['.html', 'text/html; charset=utf-8'],
  ['.js', 'text/javascript; charset=utf-8'],
  ['.css', 'text/css; charset=utf-8'],
  ['.json', 'application/json; charset=utf-8'],
  ['.txt', 'text/plain; charset=utf-8'],
  ['.md', 'text/markdown; charset=utf-8'],
  ['.svg', 'image/svg+xml'],
  ['.png', 'image/png'],
  ['.jpg', 'image/jpeg'],
  ['.jpeg', 'image/jpeg'],
  ['.webp', 'image/webp'],
  ['.ico', 'image/x-icon'],
]);

export function contentType(path) {
  return TYPES.get(extname(path).toLowerCase()) || 'application/octet-stream';
}

function safeFilePath(siteDir, requestUrl) {
  const url = new URL(requestUrl || '/', 'http://127.0.0.1');
  const decoded = decodeURIComponent(url.pathname);
  const normalized = normalize(decoded).replace(/^(\.\.(\/|\\|$))+/, '');
  const relative = normalized === '/' || normalized === '.'
    ? 'index.html'
    : normalized.replace(/^\/+/, '');
  const absolute = resolve(siteDir, relative);
  const root = resolve(siteDir);
  if (absolute !== root && !absolute.startsWith(`${root}${sep}`)) return null;
  return absolute;
}

export async function findFreePort(preferredPort) {
  for (let port = preferredPort; port < preferredPort + 50; port += 1) {
    const server = createServer();
    const available = await new Promise((resolveAvailable) => {
      server.once('error', () => resolveAvailable(false));
      server.listen(port, '127.0.0.1', () => server.close(() => resolveAvailable(true)));
    });
    if (available) return port;
  }
  throw new Error(`No free port found starting at ${preferredPort}.`);
}

export async function startStaticServer({ siteDir, port }) {
  const server = createServer((req, res) => {
    const file = safeFilePath(siteDir, req.url);
    if (!file || !existsSync(file) || !statSync(file).isFile()) {
      res.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' });
      res.end('Not found\n');
      return;
    }

    res.writeHead(200, { 'content-type': contentType(file) });
    createReadStream(file).pipe(res);
  });

  server.listen(port, '127.0.0.1');
  await once(server, 'listening');
  const address = server.address();
  const actualPort = typeof address === 'object' && address ? address.port : port;
  return {
    url: `http://127.0.0.1:${actualPort}/`,
    port: actualPort,
    close: () => new Promise((resolveClose) => server.close(resolveClose)),
  };
}

export function siteDirForRoot(root) {
  return join(root, 'site');
}
