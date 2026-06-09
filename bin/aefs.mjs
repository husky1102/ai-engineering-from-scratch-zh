#!/usr/bin/env node
import { runCli } from '../cli/main.mjs';

runCli(process.argv.slice(2), {
  argv0: process.argv[1],
  cwd: process.cwd(),
  env: process.env,
}).catch((error) => {
  const message = error && error.message ? error.message : String(error);
  console.error(`aefs: ${message}`);
  process.exitCode = 1;
});
