import { existsSync, statSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { readConfig } from './config.mjs';

const REQUIRED_MARKERS = [
  'site/index.html',
  'site/build.js',
  'phases',
  'ROADMAP.md',
  'README.md',
];

function hasPath(path) {
  try {
    return existsSync(path);
  } catch {
    return false;
  }
}

function optionValue(args, name) {
  const index = args.indexOf(name);
  if (index === -1) return null;
  return args[index + 1] || null;
}

function validRootOrThrow(root, source) {
  const absolute = resolve(root);
  if (!isProjectRoot(absolute)) {
    throw new Error(`The ${source} root is not a valid AEFS course root: ${absolute}`);
  }
  return { root: absolute, source };
}

export function isProjectRoot(candidate) {
  if (!candidate || !hasPath(candidate)) return false;
  return REQUIRED_MARKERS.every((marker) => hasPath(resolve(candidate, marker)));
}

export function findProjectRoot(start) {
  let current = resolve(start);
  while (true) {
    if (isProjectRoot(current)) return current;
    const parent = dirname(current);
    if (parent === current) return null;
    current = parent;
  }
}

export function resolveProjectRoot({
  args = [],
  cwd = process.cwd(),
  env = process.env,
  moduleUrl = import.meta.url,
} = {}) {
  const argRoot = optionValue(args, '--root');
  if (argRoot) return validRootOrThrow(argRoot, 'argument');

  if (env.AEFS_ROOT) return validRootOrThrow(env.AEFS_ROOT, 'env');

  const config = readConfig(env);
  if (config.root) return validRootOrThrow(config.root, 'config');

  const cwdRoot = findProjectRoot(cwd);
  if (cwdRoot) return { root: cwdRoot, source: 'cwd' };

  const modulePath = fileURLToPath(moduleUrl);
  const packageRoot = findProjectRoot(dirname(modulePath));
  if (packageRoot) return { root: packageRoot, source: 'package' };

  throw new Error('Cannot find an AEFS course root. Run `aefs config set-root <path>` or pass `--root <path>`.');
}

export function assertDirectory(path) {
  const stats = statSync(path);
  if (!stats.isDirectory()) throw new Error(`Expected a directory: ${path}`);
}
