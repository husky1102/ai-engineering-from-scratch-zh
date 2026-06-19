# Site Vendor Assets

Large browser runtimes are downloaded on demand and are not committed.

To make the Python browser runner work without the jsdelivr CDN, run:

```bash
scripts/vendor_pyodide.sh
```

The script mirrors Pyodide v0.26.4 into `site/vendor/pyodide/v0.26.4/full/`.
The static site serves that directory directly during local deployment, while
`.gitignore` keeps the large runtime files out of commits.
