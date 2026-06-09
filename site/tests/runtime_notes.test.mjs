import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

test('lesson runtime notes are localized before rendering', () => {
  const lessonHtml = readFileSync('site/lesson.html', 'utf8');
  const zhCn = JSON.parse(readFileSync('site/i18n/zh-CN.json', 'utf8'));
  const fallback = readFileSync('site/i18n.js', 'utf8');

  assert.match(lessonHtml, /function runtimeNoteText\(note\)/);
  assert.match(lessonHtml, /i18n\.js\?v=20260609b/);
  assert.match(lessonHtml, /'Python uses packages outside the browser allowlist': 'runtimeNoteOutsideBrowserAllowlist'/);
  assert.match(lessonHtml, /escapeHtml\(runtimeNoteText\(note\)\)/);
  assert.match(lessonHtml, /el\.classList\.remove\('panel-loading'\);/);
  assert.equal(zhCn.runtimeNoteOutsideBrowserAllowlist, 'Python 使用浏览器运行器尚未启用的包。');
  assert.match(fallback, /runtimeNoteOutsideBrowserAllowlist: 'Python 使用浏览器运行器尚未启用的包。'/);
});
