"""Stand-in for `accelerate launch` used by the runner tests.

Modes:
  replay <file> <exit_code>   write the file to stdout unchanged, then exit
  text <text> <exit_code>     write text (with \\r and \\n escapes expanded), then exit
  env <name>                  print the value of an environment variable
  sleep <pid_file>            start a sleeping child (like accelerate does), write its pid, wait
  ignore-term                 ignore SIGTERM, print "started", sleep (forces a SIGKILL)
  crash                       kill itself with SIGKILL
"""

import os
import signal
import subprocess
import sys
import time


def main(argv: list[str]) -> int:
    mode = argv[1]
    if mode == "replay":
        with open(argv[2], "rb") as source:
            sys.stdout.buffer.write(source.read())
        return int(argv[3])
    if mode == "text":
        sys.stdout.write(argv[2].replace("\\r", "\r").replace("\\n", "\n"))
        return int(argv[3])
    if mode == "env":
        print(os.environ.get(argv[2], "<unset>"))
        return 0
    if mode == "sleep":
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        with open(argv[2], "w") as handle:
            handle.write(str(child.pid))
        print("started", flush=True)
        child.wait()
        return child.returncode
    if mode == "ignore-term":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        print("started", flush=True)
        time.sleep(60)
        return 0
    if mode == "crash":
        os.kill(os.getpid(), signal.SIGKILL)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
