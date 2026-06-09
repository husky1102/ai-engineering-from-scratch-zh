import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

function loadPalette() {
  const sandbox = {
    console,
    navigator: { platform: 'MacIntel' },
    window: {},
    document: {
      readyState: 'loading',
      addEventListener() {},
      querySelectorAll() { return []; },
    },
  };
  vm.runInNewContext(readFileSync('site/cmdpalette.js', 'utf8'), sandbox);
  return sandbox.window.CmdPalette;
}

test('command palette parses search filter tokens', () => {
  const palette = loadPalette();

  const parsed = palette._test.parseQuery('linear phase:02 runtime:local-kernel status:complete lang:zh-CN');

  assert.equal(parsed.text, 'linear');
  assert.deepEqual(Object.fromEntries(Object.entries(parsed.filters)), {
    phase: '02',
    runtime: 'local-kernel',
    status: 'complete',
    lang: 'zh-cn',
  });
});
