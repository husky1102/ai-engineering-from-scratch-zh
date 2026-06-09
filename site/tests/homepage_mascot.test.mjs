import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import test from 'node:test';

const mascotPath = 'site/assets/mascot-gpt054.png';

test('homepage uses the provided mascot as the primary visual asset', () => {
  const index = readFileSync('site/index.html', 'utf8');
  const style = readFileSync('site/index.html', 'utf8');

  assert.equal(existsSync(mascotPath), true);
  assert.match(index, /<img[^>]+class="manual-mascot"[^>]+src="assets\/mascot-gpt054\.png"/);
  assert.match(index, /alt="AI 工程中文站主视觉"/);
  assert.match(style, /\.manual-mascot\s*\{[^}]*height: auto;/);
});

test('homepage favicon and social image point at the local mascot asset', () => {
  const index = readFileSync('site/index.html', 'utf8');

  assert.match(index, /<link rel="icon" type="image\/png" href="assets\/mascot-gpt054\.png">/);
  assert.match(index, /<link rel="apple-touch-icon" href="assets\/mascot-gpt054\.png">/);
  assert.match(index, /<meta property="og:image" content="assets\/mascot-gpt054\.png">/);
  assert.match(index, /<meta name="twitter:image" content="assets\/mascot-gpt054\.png">/);
});

test('shared logo icon uses the local mascot asset', () => {
  const style = readFileSync('site/style.css', 'utf8');

  assert.match(style, /\.logo-icon[\s\S]+background-image: url\('assets\/mascot-gpt054\.png'\)/);
  assert.match(style, /\.logo-icon[\s\S]+border-radius: 50%/);
});

test('top-level html pages use the local mascot favicon', () => {
  const pages = [
    'site/index.html',
    'site/catalog.html',
    'site/glossary.html',
    'site/prereqs.html',
    'site/about.html',
    'site/lesson.html',
  ];

  for (const page of pages) {
    const html = readFileSync(page, 'utf8');
    assert.match(
      html,
      /<link rel="icon" type="image\/png" href="assets\/mascot-gpt054\.png">/,
      `${page} should use the mascot favicon`,
    );
  }
});
