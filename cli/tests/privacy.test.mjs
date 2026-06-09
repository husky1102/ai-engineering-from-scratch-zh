import assert from 'node:assert/strict';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import test from 'node:test';

const SCAN_DIRS = ['bin', 'cli'];
const SECRET_PATTERN = /\b(api[_-]?key|secret|token|password|passwd|credential|bearer|cookie)\b/i;
const LOCAL_PATH_PATTERN = new RegExp(String.raw`/(Users|home|private)/(?!tmp/aefs-|var/folders/aefs-)[^\s'"` + '`' + String.raw`]+`);

function filesUnder(dir) {
  const entries = readdirSync(dir);
  return entries.flatMap((entry) => {
    const path = join(dir, entry);
    const stats = statSync(path);
    if (stats.isDirectory()) return filesUnder(path);
    return [path];
  });
}

function scannableSourceFiles() {
  return SCAN_DIRS
    .flatMap(filesUnder)
    .filter((file) => file.endsWith('.mjs'))
    .filter((file) => file !== join('cli', 'tests', 'privacy.test.mjs'));
}

test('CLI source does not contain obvious secrets or developer-local paths', () => {
  const files = scannableSourceFiles();
  assert.ok(files.length > 0);

  for (const file of files) {
    const text = readFileSync(file, 'utf8');
    assert.doesNotMatch(text, SECRET_PATTERN, file);
    assert.doesNotMatch(text, LOCAL_PATH_PATTERN, file);
  }
});

test('package metadata does not introduce install-time scripts or dependencies', () => {
  const pkg = JSON.parse(readFileSync('package.json', 'utf8'));
  assert.deepEqual(pkg.dependencies || {}, {});
  assert.deepEqual(pkg.devDependencies || {}, {});
  assert.equal(pkg.scripts?.postinstall, undefined);
  assert.equal(pkg.scripts?.preinstall, undefined);
});
