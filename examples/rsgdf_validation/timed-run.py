"""Run one workstation command, preserving its exit code and measuring RSS."""
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

output, *command = sys.argv[1:]
start = time.monotonic()
process = subprocess.Popen(command)
stop_reason = None
rss_limit = float(os.environ.get('GREEN_VALIDATION_RSS_LIMIT_GIB', '0')) * 1024**2
timeout = float(os.environ.get('GREEN_VALIDATION_TIMEOUT_SECONDS', '0'))
while process.poll() is None:
    try:
        status = Path(f'/proc/{process.pid}/status').read_text()
        rss = next(float(line.split()[1]) for line in status.splitlines() if line.startswith('VmRSS:'))
    except (FileNotFoundError, StopIteration):
        rss = 0
    if rss_limit and rss > rss_limit:
        stop_reason = 'RSS limit exceeded'
    elif timeout and time.monotonic() - start > timeout:
        stop_reason = 'Time limit exceeded'
    if stop_reason:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        break
    time.sleep(0.5)
code = process.wait()
usage = resource.getrusage(resource.RUSAGE_CHILDREN)
Path(output).write_text(json.dumps({
    "command": command, "exit_code": code, "stop_reason": stop_reason,
    "wall_seconds": time.monotonic() - start,
    "user_seconds": usage.ru_utime, "system_seconds": usage.ru_stime,
    "peak_rss_kib": usage.ru_maxrss,
}, indent=2) + "\n")
sys.exit(code if code >= 0 else 128-code)
