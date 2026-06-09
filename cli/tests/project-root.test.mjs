import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import test from 'node:test';
import { findProjectRoot, isProjectRoot, resolveProjectRoot } from '../project-root.mjs';
import { writeConfig } from '../config.mjs';

function makeCourseRoot() {
  const root = mkdtempSync(join(tmpdir(), 'aefs-root-'));
  mkdirSync(join(root, 'site'), { recursive: true });
  mkdirSync(join(root, 'phases'), { recursive: true });
  writeFileSync(join(root, 'site', 'index.html'), '<!doctype html>\n', 'utf8');
  writeFileSync(join(root, 'site', 'build.js'), 'console.log("build");\n', 'utf8');
  writeFileSync(join(root, 'ROADMAP.md'), '# Roadmap\n', 'utf8');
  writeFileSync(join(root, 'README.md'), '# Course\n', 'utf8');
  return root;
}

test('isProjectRoot requires public course markers', () => {
  const root = makeCourseRoot();
  assert.equal(isProjectRoot(root), true);
  assert.equal(isProjectRoot(dirname(root)), false);
});

test('findProjectRoot walks upward from nested directories', () => {
  const root = makeCourseRoot();
  const nested = join(root, 'phases', '01-phase', '01-lesson');
  mkdirSync(nested, { recursive: true });
  assert.equal(findProjectRoot(nested), root);
});

test('resolveProjectRoot prefers explicit --root', () => {
  const explicit = makeCourseRoot();
  const envRoot = makeCourseRoot();
  const result = resolveProjectRoot({
    args: ['--root', explicit],
    cwd: tmpdir(),
    env: { AEFS_ROOT: envRoot, HOME: mkdtempSync(join(tmpdir(), 'aefs-home-')) },
    moduleUrl: import.meta.url,
  });
  assert.equal(result.root, explicit);
  assert.equal(result.source, 'argument');
});

test('resolveProjectRoot supports AEFS_ROOT', () => {
  const root = makeCourseRoot();
  const result = resolveProjectRoot({
    args: [],
    cwd: tmpdir(),
    env: { AEFS_ROOT: root, HOME: mkdtempSync(join(tmpdir(), 'aefs-home-')) },
    moduleUrl: import.meta.url,
  });
  assert.equal(result.root, root);
  assert.equal(result.source, 'env');
});

test('resolveProjectRoot supports user config root', () => {
  const root = makeCourseRoot();
  const home = mkdtempSync(join(tmpdir(), 'aefs-home-'));
  writeConfig({ root }, { HOME: home });
  const result = resolveProjectRoot({
    args: [],
    cwd: tmpdir(),
    env: { HOME: home },
    moduleUrl: import.meta.url,
  });
  assert.equal(result.root, root);
  assert.equal(result.source, 'config');
});

test('resolveProjectRoot supports cwd upward discovery', () => {
  const root = makeCourseRoot();
  const nested = join(root, 'site', 'nested');
  mkdirSync(nested, { recursive: true });
  const result = resolveProjectRoot({
    args: [],
    cwd: nested,
    env: { HOME: mkdtempSync(join(tmpdir(), 'aefs-home-')) },
    moduleUrl: import.meta.url,
  });
  assert.equal(result.root, root);
  assert.equal(result.source, 'cwd');
});
