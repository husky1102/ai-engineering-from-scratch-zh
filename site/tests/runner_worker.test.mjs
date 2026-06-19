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
  var importScriptURL = null;
  var indexURL = null;
  var loadedPackages = null;
  var sandbox = {
    importScripts(url) {
      importScriptURL = url;
    },
    loadPyodide: async (options) => {
      indexURL = options.indexURL;
      stdout = options.stdout;
      stderr = options.stderr;
      return {
        loadPackage: async (packages) => {
          loadedPackages = packages;
        },
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
      pyodideBaseURL: 'vendor/pyodide/v0.26.4/full/',
    },
  });
  await waitForDone(messages);

  var stdoutMessage = messages.find((message) => message.type === 'stdout');
  var stderrMessage = messages.find((message) => message.type === 'stderr');
  assert.equal(stdoutMessage.runId, 'run-42');
  assert.equal(stdoutMessage.text, 'hello\n');
  assert.equal(stderrMessage.runId, 'run-42');
  assert.equal(stderrMessage.text, 'careful\n');
  assert.equal(importScriptURL, 'vendor/pyodide/v0.26.4/full/pyodide.js');
  assert.equal(indexURL, 'vendor/pyodide/v0.26.4/full/');
  assert.deepEqual(loadedPackages, ['numpy']);
});

test('worker defaults to the local vendored Pyodide path', async () => {
  var messages = [];
  var importScriptURL = null;
  var sandbox = {
    importScripts(url) {
      importScriptURL = url;
    },
    loadPyodide: async () => ({
      loadPackage: async () => {},
      runPythonAsync: async () => null,
    }),
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
      runId: 'run-default',
      code: '1 + 1',
      packages: [],
    },
  });
  await waitForDone(messages);

  assert.equal(importScriptURL, 'vendor/pyodide/v0.26.4/full/pyodide.js');
});
