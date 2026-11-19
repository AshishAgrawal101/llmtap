#!/usr/bin/env node

const { spawn, spawnSync } = require('child_process');
const path = require('path');

// Root of the npm package (contains the llmtap/ python module)
const repoRoot = path.resolve(__dirname, '..');

// Candidates to locate a Python 3.11+ executable
const candidates = [];
if (process.env.PYTHON) {
  candidates.push([process.env.PYTHON]);
}
if (process.platform === 'win32') {
  candidates.push(['python']);
  candidates.push(['py', '-3']);
  candidates.push(['python3']);
} else {
  candidates.push(['python3']);
  candidates.push(['python']);
}

function findPython() {
  for (const [cmd, ...args] of candidates) {
    try {
      const res = spawnSync(cmd, [...args, '-c', 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'], {
        stdio: 'ignore',
        windowsHide: true,
      });
      if (res.status === 0) {
        return [cmd, ...args];
      }
    } catch (e) {
      // Command not found or cannot spawn; try next candidate
    }
  }
  return null;
}

const pyBin = findPython();
if (!pyBin) {
  console.error('\x1b[31merror:\x1b[0m llmtap requires Python 3.11 or higher.');
  console.error('Please install Python 3.11+ from https://www.python.org/ or via your system package manager.');
  process.exit(1);
}

const [cmd, ...baseArgs] = pyBin;
const userArgs = process.argv.slice(2);
const pythonArgs = [...baseArgs, '-m', 'llmtap', ...userArgs];

// Ensure PYTHONPATH includes the package root so python -m llmtap is always discoverable
const env = { ...process.env };
const sep = process.platform === 'win32' ? ';' : ':';
env.PYTHONPATH = env.PYTHONPATH ? `${repoRoot}${sep}${env.PYTHONPATH}` : repoRoot;

const child = spawn(cmd, pythonArgs, {
  stdio: 'inherit',
  env,
  windowsHide: false,
});

child.on('error', (err) => {
  console.error(`\x1b[31merror:\x1b[0m failed to spawn ${cmd}: ${err.message}`);
  process.exit(1);
});

child.on('exit', (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
  } else {
    process.exit(code ?? 0);
  }
});

// Forward interruption signals to the child process
['SIGINT', 'SIGTERM', 'SIGHUP'].forEach((sig) => {
  process.on(sig, () => {
    if (child && !child.killed) {
      child.kill(sig);
    }
  });
});