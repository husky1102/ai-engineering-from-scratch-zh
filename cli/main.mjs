import { spawnSync } from 'node:child_process';
import { join, resolve } from 'node:path';
import { readConfig, writeConfig } from './config.mjs';
import { isProjectRoot, resolveProjectRoot } from './project-root.mjs';
import { findFreePort, siteDirForRoot, startStaticServer } from './site-server.mjs';
import { openBrowser, openDefaultTerminal } from './openers.mjs';

function hasFlag(args, name) {
  return args.includes(name);
}

function valueAfter(args, name) {
  const index = args.indexOf(name);
  return index === -1 ? null : args[index + 1] || null;
}

function numberOption(args, name, fallback) {
  const value = valueAfter(args, name);
  if (!value) return fallback;
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0 || parsed > 65535) {
    throw new Error(`${name} must be a TCP port between 1 and 65535.`);
  }
  return parsed;
}

function printHelp() {
  console.log(`Usage:
  aefs [--root <path>] [--port <number>] [--no-open] [--no-terminal] [--rebuild]
  learnAI [--root <path>] [--port <number>] [--no-open] [--no-terminal] [--rebuild]
  aefs doctor [--root <path>]
  aefs config set-root <path>
  aefs config get-root`);
}

function runBuild(root) {
  const result = spawnSync(process.execPath, ['site/build.js'], {
    cwd: root,
    stdio: 'inherit',
  });
  if (result.status !== 0) {
    throw new Error('site/build.js failed.');
  }
}

async function runDoctor(args, context) {
  const rootInfo = resolveProjectRoot({
    args,
    cwd: context.cwd,
    env: context.env,
    moduleUrl: import.meta.url,
  });
  const port = await findFreePort(numberOption(args, '--port', 4173));
  console.log(`Node: ${process.version}`);
  console.log(`Course root: ${rootInfo.root}`);
  console.log(`Root source: ${rootInfo.source}`);
  console.log(`Site dir: ${join(rootInfo.root, 'site')}`);
  console.log(`Suggested port: ${port}`);
}

function runConfig(args, context) {
  const action = args[1];
  if (action === 'set-root') {
    const root = args[2];
    if (!root) throw new Error('Usage: aefs config set-root <path>');
    if (!isProjectRoot(root)) throw new Error(`Not a valid AEFS course root: ${root}`);
    const current = readConfig(context.env);
    const file = writeConfig({ ...current, root: resolve(root) }, context.env);
    console.log(`Saved AEFS root to ${file}`);
    return;
  }
  if (action === 'get-root') {
    const current = readConfig(context.env);
    console.log(current.root || '');
    return;
  }
  throw new Error('Usage: aefs config set-root <path> OR aefs config get-root');
}

async function runStart(args, context) {
  const rootInfo = resolveProjectRoot({
    args,
    cwd: context.cwd,
    env: context.env,
    moduleUrl: import.meta.url,
  });
  if (hasFlag(args, '--rebuild')) runBuild(rootInfo.root);

  const preferredPort = numberOption(args, '--port', 4173);
  const port = await findFreePort(preferredPort);
  const server = await startStaticServer({ siteDir: siteDirForRoot(rootInfo.root), port });

  console.log(`AEFS site: ${server.url}`);
  console.log(`Course root: ${rootInfo.root}`);
  console.log(`Root source: ${rootInfo.source}`);
  console.log('Press Ctrl+C to stop.');

  if (!hasFlag(args, '--no-open')) openBrowser(server.url);
  if (!hasFlag(args, '--no-terminal')) openDefaultTerminal(rootInfo.root);

  await new Promise((resolveStop) => {
    process.once('SIGINT', resolveStop);
    process.once('SIGTERM', resolveStop);
  });
  await server.close();
}

export async function runCli(args, context) {
  if (args.includes('--help') || args.includes('-h')) {
    printHelp();
    return;
  }

  if (args[0] === 'config') {
    runConfig(args, context);
    return;
  }

  if (args[0] === 'doctor') {
    await runDoctor(args.slice(1), context);
    return;
  }

  await runStart(args, context);
}
