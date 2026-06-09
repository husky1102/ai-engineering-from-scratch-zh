import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';

export function configPath(env = process.env) {
  const base = env.XDG_CONFIG_HOME || (env.HOME ? join(env.HOME, '.config') : '');
  if (!base) {
    throw new Error('Cannot locate a user config directory because HOME is not set.');
  }
  return join(base, 'aefs', 'config.json');
}

export function readConfig(env = process.env) {
  const file = configPath(env);
  if (!existsSync(file)) return {};
  const parsed = JSON.parse(readFileSync(file, 'utf8'));
  return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {};
}

export function writeConfig(config, env = process.env) {
  const file = configPath(env);
  mkdirSync(dirname(file), { recursive: true });
  writeFileSync(file, `${JSON.stringify(config, null, 2)}\n`, 'utf8');
  return file;
}
