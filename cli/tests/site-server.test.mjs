import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { contentType, findFreePort, startStaticServer } from '../site-server.mjs';

test('contentType covers site assets', () => {
  assert.equal(contentType('index.html'), 'text/html; charset=utf-8');
  assert.equal(contentType('app.js'), 'text/javascript; charset=utf-8');
  assert.equal(contentType('style.css'), 'text/css; charset=utf-8');
  assert.equal(contentType('data.json'), 'application/json; charset=utf-8');
  assert.equal(contentType('mascot.png'), 'image/png');
  assert.equal(contentType('pyodide.asm.wasm'), 'application/wasm');
  assert.equal(contentType('python_stdlib.zip'), 'application/zip');
  assert.equal(contentType('numpy.whl'), 'application/zip');
});

test('findFreePort returns a usable port at or above preferred port', async () => {
  const port = await findFreePort(4173);
  assert.equal(Number.isInteger(port), true);
  assert.equal(port >= 4173, true);
});

test('startStaticServer serves index.html from a site directory', async () => {
  const root = mkdtempSync(join(tmpdir(), 'aefs-site-'));
  const siteDir = join(root, 'site');
  mkdirSync(siteDir, { recursive: true });
  writeFileSync(join(siteDir, 'index.html'), '<h1>AEFS</h1>\n', 'utf8');

  const server = await startStaticServer({ siteDir, port: 0 });
  try {
    const response = await fetch(server.url);
    assert.equal(response.status, 200);
    assert.equal(await response.text(), '<h1>AEFS</h1>\n');
  } finally {
    await server.close();
  }
});

test('startStaticServer ignores query strings when serving files', async () => {
  const root = mkdtempSync(join(tmpdir(), 'aefs-site-'));
  const siteDir = join(root, 'site');
  mkdirSync(siteDir, { recursive: true });
  writeFileSync(join(siteDir, 'lesson.html'), '<h1>Lesson</h1>\n', 'utf8');

  const server = await startStaticServer({ siteDir, port: 0 });
  try {
    const response = await fetch(`${server.url}lesson.html?path=phases/01/01`);
    assert.equal(response.status, 200);
    assert.equal(await response.text(), '<h1>Lesson</h1>\n');
  } finally {
    await server.close();
  }
});
