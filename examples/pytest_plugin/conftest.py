# Nothing to configure: the plugin ships with llmtap (pytest11 entry point).
# Without `pip install llmtap` (e.g. running from a source checkout) load it explicitly:
pytest_plugins = ["llmtap.pytest_plugin"]
