import { chmodSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { spawn } from 'node:child_process';
import { tmpdir } from 'node:os';

export function shellQuote(value) {
  return `'${String(value).replace(/'/g, `'\\''`)}'`;
}

export function browserCommand(platform, url) {
  if (platform === 'darwin') return { command: 'open', args: [url] };
  if (platform === 'win32') return { command: 'cmd', args: ['/c', 'start', '', url] };
  return { command: 'xdg-open', args: [url] };
}

export function openBrowser(url, { platform = process.platform } = {}) {
  const { command, args } = browserCommand(platform, url);
  return spawn(command, args, { stdio: 'ignore', detached: true }).unref();
}

export function createTerminalScript({ projectRoot, tempDir = tmpdir() }) {
  const file = join(tempDir, `aefs-open-repo-${process.pid}.command`);
  const content = [
    '#!/bin/zsh',
    `cd ${shellQuote(projectRoot)} || exit 1`,
    'clear',
    'printf "AEFS course root: %s\\n" "$PWD"',
    'exec "${SHELL:-/bin/zsh}" -l',
    '',
  ].join('\n');

  writeFileSync(file, content, 'utf8');
  chmodSync(file, 0o700);
  return { file, content };
}

export function openDefaultTerminal(projectRoot, { platform = process.platform, tempDir = tmpdir() } = {}) {
  if (platform !== 'darwin') {
    console.log(`Course root: ${projectRoot}`);
    console.log('Automatic new-terminal opening is currently implemented for macOS. Use --no-terminal to suppress this message.');
    return null;
  }

  const { file } = createTerminalScript({ projectRoot, tempDir });
  return spawn('open', [file], { stdio: 'ignore', detached: true }).unref();
}
