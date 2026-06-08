(function () {
  var manifestPromise = null;
  var worker = null;
  var runSeq = 0;
  var pending = Object.create(null);
  var enabledRecord = null;
  var delegatedEventsReady = false;

  function text(key) {
    var dict = (window.AIFSI18N && window.AIFSI18N.fallback && window.AIFSI18N.fallback[document.documentElement.lang]) || {};
    var en = (window.AIFSI18N && window.AIFSI18N.fallback && window.AIFSI18N.fallback.en) || {};
    return (window.__AIFS_LOCALE__ && window.__AIFS_LOCALE__[key]) || dict[key] || en[key] || key;
  }

  function loadManifest() {
    if (!manifestPromise) {
      manifestPromise = fetch('runtime-manifest.json')
        .then(function (res) {
          if (!res.ok) throw new Error('runtime-manifest-missing');
          return res.json();
        })
        .catch(function () { return null; });
    }
    return manifestPromise;
  }

  function getWorker() {
    if (worker) return worker;
    worker = new Worker('runner-worker.js?v=20260609a');
    worker.onmessage = function (event) {
      var msg = event.data || {};
      var run = pending[msg.runId];
      if (!run && msg.type !== 'ready' && msg.type !== 'stdout' && msg.type !== 'stderr') return;
      if (msg.type === 'status' && run) {
        run.status.textContent = msg.status === 'loading' ? text('runnerStarting') : text('runnerRunning');
      } else if (msg.type === 'stdout' && run) {
        append(run.output, msg.text, 'stdout');
      } else if (msg.type === 'stderr' && run) {
        append(run.output, msg.text, 'stderr');
      } else if (msg.type === 'result' && run) {
        append(run.output, msg.text, 'result');
      } else if (msg.type === 'error' && run) {
        append(run.output, msg.text + '\n', 'error');
        finishRun(run, false);
      } else if (msg.type === 'done' && run) {
        finishRun(run, true);
      }
    };
    return worker;
  }

  function resetWorker() {
    if (worker) worker.terminate();
    worker = null;
    pending = Object.create(null);
  }

  function append(output, textValue, kind) {
    if (!output) return;
    output.hidden = false;
    var span = document.createElement('span');
    span.className = 'pyodide-output-' + kind;
    span.textContent = textValue;
    output.appendChild(span);
  }

  function finishRun(run, ok) {
    clearTimeout(run.timer);
    run.button.disabled = false;
    run.button.textContent = text('run');
    run.status.textContent = ok ? text('runnerReady') : '';
    run.button.dataset.runState = ok ? 'done' : 'error';
    delete pending[run.id];
  }

  function hasClass(element, className) {
    if (!element) return false;
    if (element.classList) return element.classList.contains(className);
    return String(element.className || '').split(/\s+/).indexOf(className) !== -1;
  }

  function getRunParts(block) {
    var toolbar = block.nextElementSibling;
    if (!hasClass(toolbar, 'pyodide-toolbar')) return null;
    var output = toolbar.nextElementSibling;
    if (!hasClass(output, 'pyodide-output')) return null;
    return {
      button: toolbar.querySelector('.pyodide-run'),
      output: output,
      status: toolbar.querySelector('.pyodide-status')
    };
  }

  function runCell(block) {
    if (!enabledRecord) return;
    var code = block.getAttribute('data-code') || '';
    var parts = getRunParts(block);
    var output = parts && parts.output;
    var status = parts && parts.status;
    var button = parts && parts.button;
    if (!output || !button || !status) return;
    if (button.disabled || button.dataset.runState === 'running') return;
    output.textContent = '';
    output.hidden = false;
    status.textContent = text('runnerStarting');
    button.disabled = true;
    button.dataset.runState = 'running';
    button.textContent = text('runnerRunning');

    var id = 'run-' + (++runSeq);
    var timer = setTimeout(function () {
      append(output, text('runnerTimedOut') + '\n', 'error');
      resetWorker();
      button.disabled = false;
      button.dataset.runState = 'timeout';
      button.textContent = text('run');
      status.textContent = '';
    }, 15000);
    pending[id] = { id: id, button: button, output: output, status: status, timer: timer };
    getWorker().postMessage({
      type: 'run',
      runId: id,
      code: code,
      packages: enabledRecord.packages || []
    });
  }

  function enhanceBlocks() {
    document.querySelectorAll('pre[data-code-lang="python"], pre[data-code-lang="py"]').forEach(function (pre) {
      if (pre._pyodideEnhanced) return;
      pre._pyodideEnhanced = true;
      var toolbar = document.createElement('div');
      toolbar.className = 'pyodide-toolbar';
      var run = document.createElement('button');
      run.type = 'button';
      run.className = 'pyodide-run';
      run.dataset.runState = 'idle';
      run.setAttribute('onclick', 'window.AIFSPyodideRunner && window.AIFSPyodideRunner.runButton(this)');
      run.textContent = text('run');
      var reset = document.createElement('button');
      reset.type = 'button';
      reset.className = 'pyodide-reset';
      reset.textContent = text('resetOutput');
      var status = document.createElement('span');
      status.className = 'pyodide-status';
      toolbar.appendChild(run);
      toolbar.appendChild(reset);
      toolbar.appendChild(status);
      var output = document.createElement('pre');
      output.className = 'pyodide-output';
      output.hidden = true;
      pre.insertAdjacentElement('afterend', output);
      pre.insertAdjacentElement('afterend', toolbar);
      reset.addEventListener('click', function () {
        output.textContent = '';
        output.hidden = true;
        status.textContent = '';
        run.dataset.runState = 'idle';
      });
    });
  }

  function initDelegatedEvents() {
    if (delegatedEventsReady) return;
    delegatedEventsReady = true;
    document.addEventListener('click', function (event) {
      var target = event.target;
      var runButton = target && target.closest ? target.closest('.pyodide-run') : null;
      if (!runButton) return;
      var toolbar = runButton.closest('.pyodide-toolbar');
      var block = toolbar ? toolbar.previousElementSibling : null;
      if (block && block.matches('pre[data-code-lang="python"], pre[data-code-lang="py"]')) {
        runCell(block);
      }
    });
  }

  function runButton(button) {
    var toolbar = button ? button.closest('.pyodide-toolbar') : null;
    var block = toolbar ? toolbar.previousElementSibling : null;
    if (block && block.matches('pre[data-code-lang="python"], pre[data-code-lang="py"]')) {
      runCell(block);
    }
  }

  function init(lessonPath) {
    return loadManifest().then(function (manifest) {
      var record = manifest && manifest.lessons ? manifest.lessons[lessonPath] : null;
      if (!record || record.runtime !== 'browser-pyodide') return false;
      enabledRecord = record;
      initDelegatedEvents();
      enhanceBlocks();
      return true;
    });
  }

  window.AIFSPyodideRunner = { init: init, resetWorker: resetWorker, runButton: runButton };
})();
