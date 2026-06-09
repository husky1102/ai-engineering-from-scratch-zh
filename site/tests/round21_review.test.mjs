import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

function read(file) {
  return readFileSync(file, 'utf8');
}

test('catalog page is Chinese and links lessons to the local reader', () => {
  const html = read('site/catalog.html');

  assert.match(html, /<h1>课程目录<\/h1>/);
  assert.match(html, /placeholder="搜索课程/);
  assert.match(html, />全部阶段</);
  assert.match(html, />全部状态</);
  assert.match(html, />已完成</);
  assert.match(html, />计划中</);
  assert.match(html, />阶段/);
  assert.match(html, />课程/);
  assert.match(html, />类型/);
  assert.match(html, />语言/);
  assert.match(html, />状态/);
  assert.match(html, /function lessonPathFromUrl\(url\)/);
  assert.match(html, /function localLessonHref\(url\)/);
  assert.match(html, /localLessonHref\(r\.url\)/);
  assert.doesNotMatch(html, /target="_blank" rel="noopener">'\s*\+\s*escapeHtml\(r\.name\)/);
});

test('shared logo icon keeps the image but removes framed chrome', () => {
  const style = read('site/style.css');

  assert.match(style, /\.logo-icon[\s\S]+background-image: url\('assets\/mascot-gpt054\.png'\)/);
  assert.match(style, /\.logo-icon[\s\S]+border: 0;/);
  assert.match(style, /\.logo-icon[\s\S]+box-shadow: none;/);
});

test('command palette user-facing chrome is Chinese', () => {
  const palette = read('site/cmdpalette.js');

  assert.match(palette, /aria-label', '搜索课程和术语'/);
  assert.match(palette, /placeholder="搜索课程和术语…"/);
  assert.match(palette, /输入关键词，或使用 phase:02 runtime:local-kernel status:complete 筛选/);
  assert.match(palette, /没有找到/);
  assert.match(palette, />导航</);
  assert.match(palette, />打开</);
  assert.match(palette, />关闭</);
  assert.match(palette, /第 '\s*\+\s*String\(r\.phaseId\)\.padStart\(2, '0'\)\s*\+\s*' 阶段'/);
  assert.match(palette, /术语/);
});

test('secondary pages have Chinese page chrome and body copy', () => {
  const prereqs = read('site/prereqs.html');
  const glossary = read('site/glossary.html');
  const about = read('site/about.html');

  assert.match(prereqs, /<h1>学习路线图<\/h1>/);
  assert.match(prereqs, /点击任一阶段/);
  assert.match(prereqs, />已完成</);
  assert.match(prereqs, />进行中</);
  assert.match(prereqs, />计划中</);
  assert.match(prereqs, /清除选择/);
  assert.match(prereqs, /滚动查看完整图谱/);
  assert.doesNotMatch(prereqs, /View on GitHub|Complete|In Progress|Planned|Clear selection/);

  assert.match(glossary, /<h1>术语表<\/h1>/);
  assert.match(glossary, /搜索术语/);
  assert.match(glossary, /常见说法/);
  assert.match(glossary, /实际含义/);
  assert.match(glossary, /没有匹配的术语/);

  assert.match(about, /<h1>关于这个项目<\/h1>/);
  assert.match(about, /中文本地阅读站/);
  assert.match(about, /从零开始的 AI 工程/);
  assert.match(about, /查看原仓库/);
  assert.doesNotMatch(about, /About this project|Why it exists|Who builds it|Maintained by Rohit Ghumare|Open an issue/);
});

test('home contents and modal consistently prefer Chinese phase and lesson names', () => {
  const app = read('site/app.js');

  assert.match(app, /var phaseName = p\.nameZh \|\| p\.name;/);
  assert.match(app, /var phaseDesc = p\.descZh \|\| p\.desc;/);
  assert.match(app, /var title = l\.nameZh \|\| l\.name;/);
  assert.match(app, /modalTitle'\)\.textContent = phaseName/);
  assert.match(app, /modalDesc'\)\.textContent = phaseDesc/);
});

test('lesson sidebar and code-card actions handle localized names and failures', () => {
  const lesson = read('site/lesson.html');

  assert.match(lesson, /function phaseDisplayName\(phase\)/);
  assert.match(lesson, /phase\.nameZh \|\| phase\.name/);
  assert.match(lesson, /function lessonDisplayName\(les\)/);
  assert.match(lesson, /les\.nameZh \|\| les\.name/);
  assert.match(lesson, /function sourceHref\(file\)/);
  assert.match(lesson, /href="' \+ escapeAttr\(sourceHref\(file\)\) \+ '"/);
  assert.match(lesson, /function copyText\(text\)/);
  assert.match(lesson, /t\('copyFailed'\)/);
});
