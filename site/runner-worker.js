/* global importScripts, loadPyodide */
(function () {
  var pyodide = null;
  var loading = null;
  var currentRunId = null;
  var runtimeBaseURL = null;
  var DEFAULT_PYODIDE_BASE_URL = 'vendor/pyodide/v0.26.4/full/';

  function post(type, payload) {
    self.postMessage(Object.assign({ type: type }, payload || {}));
  }

  function withTrailingSlash(value) {
    return String(value || '').replace(/\/?$/, '/');
  }

  function loadRuntime(packages, requestedBaseURL) {
    var nextBaseURL = withTrailingSlash(requestedBaseURL || DEFAULT_PYODIDE_BASE_URL);
    if (runtimeBaseURL && runtimeBaseURL !== nextBaseURL) {
      throw new Error('Pyodide is already loaded from ' + runtimeBaseURL);
    }
    if (!loading) {
      runtimeBaseURL = nextBaseURL;
      loading = (async function () {
        importScripts(runtimeBaseURL + 'pyodide.js');
        pyodide = await loadPyodide({
          indexURL: runtimeBaseURL,
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
        await loadRuntime(msg.packages || [], msg.pyodideBaseURL);
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
