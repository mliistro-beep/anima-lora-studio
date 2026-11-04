#!/usr/bin/env bash
# Start the app from the source tree, for development only.
#
# Not the release launcher: users start the app with ./run_linux.sh (which also installs the menu
# entry). This one adds the dev tools (--setup) and the scratch folders (--scratch).
#
# Usage: scripts/dev-run.sh [--setup] [--scratch [DIR]] [-- QT_ARGS...]
#   --setup          create .venv and install the app with the dev tools if .venv is missing
#   --scratch [DIR]  use DIR for the data folder and settings (XDG_DATA_HOME, XDG_CONFIG_HOME),
#                    so your own setup is not touched; default DIR is ~/anima-dev-scratch.
#                    Not under /tmp: the training runtime and models take over 10 GB.
#   -h, --help       show this help

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPO_DIR
readonly VENV_DIR="$REPO_DIR/.venv"
readonly APP="$VENV_DIR/bin/anima-trainer"
readonly DEFAULT_SCRATCH="$HOME/anima-dev-scratch"
readonly MIN_PYTHON="3.12"

usage() {
    sed -n '/^# Usage:/,/^$/{/^$/!p}' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

setup_venv() {
    if [[ -x "$APP" ]]; then
        echo "INFO: $VENV_DIR already set up."
        return
    fi
    command -v python3 >/dev/null || die "python3 not found."
    python3 -c "import sys; sys.exit(sys.version_info < (${MIN_PYTHON/./, }))" \
        || die "Python $MIN_PYTHON or newer is needed, found $(python3 --version)."
    echo "INFO: Creating $VENV_DIR and installing the app with the dev tools..."
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install -e "$REPO_DIR[dev]"
}

setup=false
scratch=""
while (($#)); do
    case "$1" in
        --setup) setup=true ;;
        --scratch)
            if [[ $# -gt 1 && "$2" != -* ]]; then
                scratch="$2"
                shift
            else
                scratch="$DEFAULT_SCRATCH"
            fi
            ;;
        -h | --help) usage; exit 0 ;;
        --) shift; break ;;
        *) usage >&2; die "Unknown option: $1" ;;
    esac
    shift
done

if $setup; then
    setup_venv
fi
[[ -x "$APP" ]] || die "$APP not found. Run: scripts/dev-run.sh --setup"

if [[ -n "$scratch" ]]; then
    mkdir -p "$scratch/share" "$scratch/config"
    scratch="$(cd "$scratch" && pwd)"
    export XDG_DATA_HOME="$scratch/share" XDG_CONFIG_HOME="$scratch/config"
    echo "INFO: Scratch setup in $scratch (data folder $XDG_DATA_HOME/anima-trainer)."
fi

cd "$REPO_DIR"
exec "$APP" "$@"
