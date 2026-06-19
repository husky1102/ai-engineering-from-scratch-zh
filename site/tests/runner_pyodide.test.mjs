import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

class Element {
  constructor(tagName) {
    this.tagName = tagName.toUpperCase();
    this.attributes = Object.create(null);
    this.children = [];
    this.dataset = Object.create(null);
    this.parentElement = null;
    this.nextElementSibling = null;
    this.previousElementSibling = null;
    this.className = '';
    this.textContent = '';
    this.hidden = false;
    this.disabled = false;
  }

  setAttribute(name, value) {
    this.attributes[name] = String(value);
  }

  getAttribute(name) {
    return this.attributes[name] || '';
  }

  appendChild(child) {
    var last = this.children[this.children.length - 1] || null;
    if (last) {
      last.nextElementSibling = child;
      child.previousElementSibling = last;
    }
    child.parentElement = this;
    this.children.push(child);
    return child;
  }

  insertAdjacentElement(position, element) {
    assert.equal(position, 'afterend');
    var oldNext = this.nextElementSibling;
    this.nextElementSibling = element;
    element.previousElementSibling = this;
    element.nextElementSibling = oldNext;
    if (oldNext) oldNext.previousElementSibling = element;
    element.parentElement = this.parentElement;
    return element;
  }

  addEventListener() {}

  closest(selector) {
    var current = this;
    while (current) {
      if (current.matches(selector)) return current;
      current = current.parentElement;
    }
    return null;
  }

  matches(selector) {
    if (selector.startsWith('.')) return hasClass(this, selector.slice(1));
    if (selector === 'pre[data-code-lang="python"], pre[data-code-lang="py"]') {
      return this.tagName === 'PRE' && ['python', 'py'].includes(this.getAttribute('data-code-lang'));
    }
    return false;
  }

  querySelector(selector) {
    for (var child of this.children) {
      if (child.matches(selector)) return child;
      var found = child.querySelector(selector);
      if (found) return found;
    }
    return null;
  }
}

function hasClass(element, className) {
  return String(element.className).split(/\s+/).includes(className);
}

async function loadRunner() {
  var lessonPath = 'phases/01-math-foundations/01-linear-algebra-intuition';
  var pre = new Element('pre');
  pre.setAttribute('data-code-lang', 'python');
  pre.setAttribute('data-code', 'print("hello")');
  var workerMessages = [];

  class FakeWorker {
    constructor() {
      this.onmessage = null;
    }

    postMessage(message) {
      workerMessages.push(message);
    }

    terminate() {}
  }

  var sandbox = {
    Worker: FakeWorker,
    clearTimeout() {},
    console,
    document: {
      documentElement: { lang: 'en' },
      addEventListener() {},
      createElement(tagName) { return new Element(tagName); },
      querySelectorAll(selector) {
        assert.equal(selector, 'pre[data-code-lang="python"], pre[data-code-lang="py"]');
        return [pre];
      },
    },
    fetch(url) {
      assert.equal(url, 'runtime-manifest.json');
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({
          lessons: {
            [lessonPath]: {
              runtime: 'browser-pyodide',
              packages: ['numpy'],
            },
          },
        }),
      });
    },
    setTimeout() {
      return 1;
    },
    window: {
      AIFSI18N: {
        fallback: {
          en: {
            run: 'Run',
            runnerRunning: 'Running...',
            runnerStarting: 'Starting...',
          },
        },
      },
    },
  };

  vm.runInNewContext(readFileSync('site/runner-pyodide.js', 'utf8'), sandbox);
  await sandbox.window.AIFSPyodideRunner.init(lessonPath);
  return { pre, runner: sandbox.window.AIFSPyodideRunner, workerMessages };
}

test('runButton starts a code cell whose toolbar and output are sibling elements', async () => {
  var fixture = await loadRunner();
  var toolbar = fixture.pre.nextElementSibling;
  var output = toolbar.nextElementSibling;
  var runButton = toolbar.querySelector('.pyodide-run');

  fixture.runner.runButton(runButton);

  assert.equal(runButton.dataset.runState, 'running');
  assert.equal(runButton.disabled, true);
  assert.equal(output.hidden, false);
  assert.deepEqual(JSON.parse(JSON.stringify(fixture.workerMessages)), [{
    type: 'run',
    runId: 'run-1',
    code: 'print("hello")',
    packages: ['numpy'],
    pyodideBaseURL: 'vendor/pyodide/v0.26.4/full/',
  }]);
});

test('runButton ignores a button that is already running', async () => {
  var fixture = await loadRunner();
  var toolbar = fixture.pre.nextElementSibling;
  var runButton = toolbar.querySelector('.pyodide-run');

  fixture.runner.runButton(runButton);
  fixture.runner.runButton(runButton);

  assert.equal(runButton.dataset.runState, 'running');
  assert.equal(fixture.workerMessages.length, 1);
});
