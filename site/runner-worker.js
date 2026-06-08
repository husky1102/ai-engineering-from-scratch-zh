/* global importScripts, loadPyodide */
(function () {
  var pyodide = null;
  var loading = null;
  var currentRunId = null;

  function post(type, payload) {
    self.postMessage(Object.assign({ type: type }, payload || {}));
  }

  function loadRuntime(packages) {
    if (!loading) {
      loading = (async function () {
        importScripts('https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js');
        pyodide = await loadPyodide({
          stdout: function (text) { post('stdout', { runId: currentRunId, text: String(text) + '\n' }); },
          stderr: function (text) { post('stderr', { runId: currentRunId, text: String(text) + '\n' }); }
        });
        post('ready');
      })();
    }
    return loading.then(async function () {
      var allowed = Array.isArray(packages) ? packages.filter(function (pkg) { return pkg === 'numpy'; }) : [];
      if (allowed.length) await pyodide.loadPackage(allowed);
    });
  }

  self.onmessage = function (event) {
    var msg = event.data || {};
    if (msg.type !== 'run') return;
    (async function () {
      try {
        post('status', { runId: msg.runId, status: 'loading' });
        await loadRuntime(msg.packages || []);
        post('status', { runId: msg.runId, status: 'running' });
        currentRunId = msg.runId;
        var result = await pyodide.runPythonAsync(String(msg.code || ''));
        currentRunId = null;
        if (result !== undefined && result !== null) {
          post('result', { runId: msg.runId, text: String(result) + '\n' });
        }
        post('done', { runId: msg.runId });
      } catch (err) {
        currentRunId = null;
        post('error', { runId: msg.runId, text: err && err.message ? err.message : String(err) });
      }
    })();
  };
})();
