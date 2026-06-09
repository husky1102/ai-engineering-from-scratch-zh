import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import test from 'node:test';
import { runCli } from '../main.mjs';

function makeCourseRoot() {
  const root = mkdtempSync(join(tmpdir(), 'aefs-main-root-'));
  mkdirSync(join(root, 'site'), { recursive: true });
  mkdirSync(join(root, 'phases'), { recursive: true });
  writeFileSync(join(root, 'site', 'index.html'), '<!doctype html>\n', 'utf8');
  writeFileSync(join(root, 'site', 'build.js'), 'console.log("build");\n', 'utf8');
  writeFileSync(join(root, 'ROADMAP.md'), '# Roadmap\n', 'utf8');
  writeFileSync(join(root, 'README.md'), '# Course\n', 'utf8');
  return root;
}

test('config set-root persists an absolute project root', async () => {
  const root = makeCourseRoot();
  const home = mkdtempSync(join(tmpdir(), 'aefs-main-home-'));
  const originalLog = console.log;
  console.log = () => {};

  try {
    await runCli(['config', 'set-root', root], {
      argv0: 'aefs',
      cwd: tmpdir(),
      env: { HOME: home },
    });
  } finally {
    console.log = originalLog;
  }

  const config = JSON.parse(readFileSync(join(home, '.config', 'aefs', 'config.json'), 'utf8'));
  assert.equal(config.root, resolve(root));
});
