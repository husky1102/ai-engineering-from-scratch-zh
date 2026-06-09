import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { configPath, readConfig, writeConfig } from '../config.mjs';

test('configPath prefers XDG_CONFIG_HOME', () => {
  const base = mkdtempSync(join(tmpdir(), 'aefs-config-'));
  const result = configPath({ XDG_CONFIG_HOME: base, HOME: '/unused-home' });
  assert.equal(result, join(base, 'aefs', 'config.json'));
});

test('configPath falls back to HOME .config', () => {
  const home = mkdtempSync(join(tmpdir(), 'aefs-home-'));
  const result = configPath({ HOME: home });
  assert.equal(result, join(home, '.config', 'aefs', 'config.json'));
});

test('readConfig returns empty object when config is missing', () => {
  const home = mkdtempSync(join(tmpdir(), 'aefs-home-'));
  assert.deepEqual(readConfig({ HOME: home }), {});
});

test('writeConfig creates user config outside the repository', () => {
  const home = mkdtempSync(join(tmpdir(), 'aefs-home-'));
  const file = writeConfig({ root: '/example/course' }, { HOME: home });
  assert.equal(file, join(home, '.config', 'aefs', 'config.json'));
  assert.deepEqual(JSON.parse(readFileSync(file, 'utf8')), { root: '/example/course' });
});
