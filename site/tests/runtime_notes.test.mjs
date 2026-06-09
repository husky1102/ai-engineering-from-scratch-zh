import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

test('lesson runtime notes describe local execution, not browser running', () => {
  const lessonHtml = readFileSync('site/lesson.html', 'utf8');
  const zhCn = JSON.parse(readFileSync('site/i18n/zh-CN.json', 'utf8'));
  const fallback = readFileSync('site/i18n.js', 'utf8');

  assert.match(lessonHtml, /function runtimeNoteText\(note\)/);
  assert.match(lessonHtml, /i18n\.js\?v=20260609r19/);
  assert.match(lessonHtml, /function normalizeRuntime\(runtime\)/);
  assert.match(lessonHtml, /function normalizeRuntimeNotes\(runtime, notes\)/);
  assert.match(lessonHtml, /escapeHtml\(runtimeNoteText\(note\)\)/);
  assert.match(lessonHtml, /el\.classList\.remove\('panel-loading'\);/);
  assert.equal(zhCn.runtimeLocalNote, '这节课需要本地 Python 环境或 notebook kernel。');
  assert.match(fallback, /runtimeLocalNote: '这节课需要本地 Python 环境或 notebook kernel。'/);
  assert.doesNotMatch(lessonHtml, /runner-pyodide\.js/);
  assert.doesNotMatch(fallback, /runtimeBrowser|runnerStarting|runnerRunning|resetOutput/);
});

test('code panel subtitle leaves the loading state after manifests resolve', () => {
  const lessonHtml = readFileSync('site/lesson.html', 'utf8');

  assert.match(lessonHtml, /var codePanelSubtitle = panel\.querySelector\('\.ai-panel-subtitle'\);/);
  assert.match(
    lessonHtml,
    /codePanelSubtitle\.textContent = record \? runtimeLabel\(normalizeRuntime\(record\.runtime \|\| 'static-only'\)\) : \(codeFiles\.length \? t\('sourceFiles'\) : t\('runtimeStatic'\)\);/,
  );
});
