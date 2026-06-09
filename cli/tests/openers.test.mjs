import assert from 'node:assert/strict';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { browserCommand, createTerminalScript, shellQuote } from '../openers.mjs';

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

test('shellQuote handles spaces and single quotes', () => {
  assert.equal(shellQuote('/tmp/aefs course'), "'/tmp/aefs course'");
  assert.equal(shellQuote("/tmp/aefs'course"), "'/tmp/aefs'\\''course'");
});

test('browserCommand uses platform default opener', () => {
  assert.deepEqual(browserCommand('darwin', 'http://127.0.0.1:4173/'), {
    command: 'open',
    args: ['http://127.0.0.1:4173/'],
  });
  assert.deepEqual(browserCommand('linux', 'http://127.0.0.1:4173/'), {
    command: 'xdg-open',
    args: ['http://127.0.0.1:4173/'],
  });
});

test('createTerminalScript writes dynamic project root into temp script', () => {
  const temp = mkdtempSync(join(tmpdir(), 'aefs-terminal-'));
  const root = join(temp, 'course root');
  const result = createTerminalScript({ projectRoot: root, tempDir: temp });

  assert.match(result.file, /aefs-open-repo/);
  assert.match(result.content, new RegExp(`cd ${escapeRegExp(shellQuote(root))}`));
  assert.doesNotMatch(result.content, /<repo-root>/);
});
