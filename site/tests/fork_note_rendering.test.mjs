import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const lesson = readFileSync('site/lesson.html', 'utf8');

test('lesson renderer supports visible fork-note blocks', () => {
  assert.match(lesson, /fork-note/);
  assert.match(lesson, /revision-note/);
  assert.match(lesson, /GPT 新增修订，仅供参考/);
});

test('lesson renderer shows quiz provenance from _meta', () => {
  assert.match(lesson, /_meta/);
  assert.match(lesson, /quiz-provenance/);
  assert.match(lesson, /GPT 新增修订，仅供参考/);
});
