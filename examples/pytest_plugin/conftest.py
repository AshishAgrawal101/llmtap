# Nothing to configure: the plugin ships with llmtap (pytest11 entry point).
# Without `pip install llmtap-proxy` (e.g. running from a source checkout) there is no entry point, so load it
# explicitly. Only then: loading it next to the entry point's copy makes pytest refuse to start.
import os
from importlib.metadata import entry_points

if os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") or not any(
        ep.value == "llmtap.pytest_plugin" for ep in entry_points(group="pytest11")):
    pytest_plugins = ["llmtap.pytest_plugin"]
