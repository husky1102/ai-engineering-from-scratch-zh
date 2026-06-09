import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

test('homepage uses Chinese navigation and keeps only one original-repo button', () => {
  const index = readFileSync('site/index.html', 'utf8');

  assert.match(index, />目录</);
  assert.match(index, />课程</);
  assert.match(index, />路线图</);
  assert.match(index, />术语表</);
  assert.match(index, />关于</);
  assert.match(index, />查看原仓库</);

  assert.doesNotMatch(index, /class="header-github"/);
  assert.doesNotMatch(index, /data-gh-stars=/);
  assert.doesNotMatch(index, /Star on GitHub/);
  assert.doesNotMatch(index, /Follow @rohitg00/);
  assert.doesNotMatch(index, /Maintained by Rohit Ghumare/);
  assert.doesNotMatch(index, /30\.4K/);

  const originalRepoMatches = index.match(/https:\/\/github\.com\/rohitg00\/ai-engineering-from-scratch/g) || [];
  assert.equal(originalRepoMatches.length, 1);
});

test('shared static pages remove star chrome and use Chinese navigation labels', () => {
  const pages = ['site/index.html', 'site/lesson.html', 'site/catalog.html', 'site/glossary.html', 'site/prereqs.html', 'site/about.html'];

  for (const file of pages) {
    const html = readFileSync(file, 'utf8');
    assert.match(html, />目录</, file);
    assert.match(html, />课程</, file);
    assert.match(html, />路线图</, file);
    assert.match(html, />术语表</, file);
    assert.match(html, />关于</, file);
    assert.doesNotMatch(html, /class="header-github"/, file);
    assert.doesNotMatch(html, /data-gh-stars=/, file);
    assert.doesNotMatch(html, /header\.js/, file);
    assert.doesNotMatch(html, /30\.4K/, file);
  }
});

test('homepage modal chrome is Chinese and lesson title links are local', () => {
  const index = readFileSync('site/index.html', 'utf8');
  const app = readFileSync('site/app.js', 'utf8');

  assert.match(index, /进度仅保存在本浏览器/);
  assert.match(index, /重置进度/);
  assert.match(app, /function lessonUrl\(lessonPath\)/);
  assert.match(app, /lessonUrl\(lessonPath\)/);
  assert.doesNotMatch(app, /'<a href="' \+ l\.url \+ '" target="_blank"/);
  assert.match(app, /第 ' \+ String\(p\.id\)\.padStart\(2, '0'\) \+ ' 阶段'/);
  assert.match(app, /已完成/);
  assert.match(app, /阅读/);
  assert.match(app, /复习/);
  assert.match(app, /标记完成/);
});

test('build data exposes Chinese phase and lesson titles for homepage and lesson sidebar', () => {
  const build = readFileSync('site/build.js', 'utf8');
  const lesson = readFileSync('site/lesson.html', 'utf8');

  assert.match(build, /const PHASE_ZH = {/);
  assert.match(build, /function readLocalizedLessonTitle/);
  assert.match(build, /nameZh:/);
  assert.match(build, /descZh:/);
  assert.match(lesson, /phase\.nameZh\s*\|\|\s*phase\.name/);
  assert.match(lesson, /les\.nameZh\s*\|\|\s*les\.name/);
  assert.match(lesson, /第 '\s*\+\s*String\(/);
});

test('lesson page no longer exposes browser code runner UI', () => {
  const lesson = readFileSync('site/lesson.html', 'utf8');
  const i18n = readFileSync('site/i18n.js', 'utf8');
  const palette = readFileSync('site/cmdpalette.js', 'utf8');

  assert.doesNotMatch(lesson, /runner-pyodide\.js/);
  assert.doesNotMatch(lesson, /data-run-state/);
  assert.doesNotMatch(lesson, /浏览器可运行|Browser-ready Python/);
  assert.doesNotMatch(i18n, /runtimeBrowser|runnerStarting|runnerRunning|resetOutput/);
  assert.doesNotMatch(palette, /runtime:browser-pyodide|browser-pyodide/);
});
