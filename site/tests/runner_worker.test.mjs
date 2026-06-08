import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

async function waitForDone(messages) {
  for (var i = 0; i < 20; i += 1) {
    if (messages.some((message) => message.type === 'done')) return;
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
}

test('worker tags stdout and stderr with the active run id', async () => {
  var messages = [];
  var stdout = null;
  var stderr = null;
  var sandbox = {
    importScripts() {},
    loadPyodide: async (options) => {
      stdout = options.stdout;
      stderr = options.stderr;
      return {
        loadPackage: async () => {},
        runPythonAsync: async () => {
          stdout('hello');
          stderr('careful');
          return null;
        },
      };
    },
    self: {
      postMessage(message) {
        messages.push(message);
      },
    },
  };

  vm.runInNewContext(readFileSync('site/runner-worker.js', 'utf8'), sandbox);
  sandbox.self.onmessage({
    data: {
      type: 'run',
      runId: 'run-42',
      code: 'print("hello")',
      packages: ['numpy', 'torch'],
    },
  });
  await waitForDone(messages);

  var stdoutMessage = messages.find((message) => message.type === 'stdout');
  var stderrMessage = messages.find((message) => message.type === 'stderr');
  assert.equal(stdoutMessage.runId, 'run-42');
  assert.equal(stdoutMessage.text, 'hello\n');
  assert.equal(stderrMessage.runId, 'run-42');
  assert.equal(stderrMessage.text, 'careful\n');
});
