#!/usr/bin/env bash
# Start Anima LoRA Studio from this checkout (Linux).
#
# The first start creates .venv next to this script and installs the app there (it needs the
# network); later starts reinstall only when pyproject.toml changed (for example after a
# git pull) or the checkout was moved. The training runtime (torch, sd-scripts, models) is
# installed by the app itself, from the Settings tab, into its data folder.
#
# Usage: ./run_linux.sh [--install-desktop | --uninstall-desktop] [APP_ARGS...]
#   --install-desktop    set up .venv, then add Anima LoRA Studio to the applications menu
#   --uninstall-desktop  remove the menu entry and its icon (.venv and the data stay)
#   -h, --help           show this help

set -euo pipefail

# Resolved, so a symlink to this script (for example in ~/bin) still finds the checkout.
LAUNCHER="$(readlink -f "${BASH_SOURCE[0]}")"
readonly LAUNCHER
REPO_DIR="$(dirname "$LAUNCHER")"
readonly REPO_DIR
readonly VENV_DIR="$REPO_DIR/.venv"
readonly VENV_PYTHON="$VENV_DIR/bin/python"
readonly APP="$VENV_DIR/bin/anima-trainer"
readonly STAMP="$VENV_DIR/.anima-pyproject.sha256"
readonly MIN_PYTHON="3.12"
readonly PYTHON_CANDIDATES=(python3 python3.14 python3.13 python3.12)
# Same name as anima_trainer.ui.app_icon.DESKTOP_FILE_NAME (setDesktopFileName).
readonly APP_ID="anima-lora-studio"
readonly APP_TITLE="Anima LoRA Studio"
readonly ICON_SOURCE="$REPO_DIR/src/anima_trainer/ui/assets/$APP_ID.svg"
readonly DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
readonly DESKTOP_ENTRY="$DATA_HOME/applications/$APP_ID.desktop"
readonly ICON_TARGET="$DATA_HOME/icons/hicolor/scalable/apps/$APP_ID.svg"
# Started from the menu there is no terminal: the launcher's own output goes to this log too.
readonly LOG_FILE="${XDG_STATE_HOME:-$HOME/.local/state}/anima-trainer/launcher.log"
logging=false

usage() {
    sed -n '/^# Usage:/,/^$/{/^$/!p}' "$LAUNCHER" | sed 's/^# \{0,1\}//'
}

# Without a terminal: copy stdout and stderr to the log (fds 3 and 4 keep the originals, so
# the app itself does not write to the log).
start_log() {
    [[ -t 2 ]] && return
    mkdir -p "$(dirname "$LOG_FILE")" || return 0
    exec 3>&1 4>&2
    exec > >(tee -a "$LOG_FILE") 2> >(tee -a "$LOG_FILE" >&2)
    logging=true
    echo "INFO: $(printf '%(%F %T)T' -1) $LAUNCHER $*"
}

stop_log() {
    if $logging; then
        exec 1>&3 2>&4 3>&- 4>&-
        logging=false
    fi
}

# A desktop notification, only without a terminal and only if notify-send exists.
notify() {
    if $logging && command -v notify-send >/dev/null; then
        notify-send --app-name="$APP_TITLE" "$APP_TITLE" "$1" || true
    fi
}

die() {
    echo "ERROR: $*" >&2
    notify "$*"
    exit 1
}

find_python() {
    local candidate
    for candidate in "${PYTHON_CANDIDATES[@]}"; do
        if command -v "$candidate" >/dev/null \
            && "$candidate" -c "import sys; sys.exit(sys.version_info < (${MIN_PYTHON/./, }))"; then
            echo "$candidate"
            return
        fi
    done
    local found="none"
    if command -v python3 >/dev/null; then
        found="$(python3 --version 2>&1)"
    fi
    die "Python $MIN_PYTHON or newer is needed (found: $found). Install it with your package" \
        "manager, for example: sudo apt install python3 python3-venv"
}

# What the stamp records: pyproject.toml and the checkout path (a moved checkout must reinstall,
# because the app script in .venv points to the old path).
install_key() {
    local out
    out="$(sha256sum "$REPO_DIR/pyproject.toml")"
    echo "${out%% *} $REPO_DIR"
}

# Called only when .venv does not exist, so removing it on failure removes only what this made.
create_venv() {
    local python
    python="$(find_python)"
    echo "INFO: Creating $VENV_DIR with $python..."
    trap 'rm -rf "$VENV_DIR"; exit 130' INT TERM
    if ! "$python" -m venv "$VENV_DIR" || ! "$VENV_PYTHON" -m pip --version >/dev/null 2>&1; then
        rm -rf "$VENV_DIR"
        die "Could not create $VENV_DIR with pip (on Debian/Ubuntu: sudo apt install python3-venv)."
    fi
    trap - INT TERM
}

# Create .venv if needed and (re)install the app when the install key changed since the last
# install. The development .venv (scripts/dev-run.sh --setup) is reused as it is.
setup_venv() {
    [[ -e "$VENV_DIR" ]] || create_venv
    "$VENV_PYTHON" -m pip --version >/dev/null 2>&1 \
        || die "$VENV_DIR has no working Python or pip (was the system Python upgraded?)." \
            "Remove $VENV_DIR and start again."
    local wanted
    wanted="$(install_key)"
    if [[ -x "$APP" && -f "$STAMP" && "$(cat "$STAMP")" == "$wanted" ]]; then
        return
    fi
    echo "INFO: Installing the app into $VENV_DIR (this needs the network)..."
    notify "Installing the app; the window opens when it is done (a few minutes the first time)."
    rm -f "$STAMP"
    if "$VENV_PYTHON" -m pip install -e "$REPO_DIR"; then
        echo "$wanted" > "$STAMP"
        return
    fi
    if [[ -x "$APP" ]]; then
        echo "WARN: Updating the app failed (see the pip output above); starting the installed" \
            "version. The update is retried at the next start." >&2
        return
    fi
    die "Installing the app failed; see the pip output above. It is retried at the next start."
}

# One install at a time (two quick clicks on the menu entry); fd 9 is closed before the app runs.
locked_setup_venv() {
    if command -v flock >/dev/null; then
        exec 9<"$LAUNCHER"
        flock 9
    fi
    setup_venv
    exec 9<&-
}

# The Exec value is written in double quotes; these characters would need escaping there.
check_exec_path() {
    case "$LAUNCHER" in
        *[\"\`\$\\%]* | *$'\n'*)
            die "The checkout path cannot be used in a menu entry: $LAUNCHER" \
                "(move the checkout to a folder without \" \` \$ \\ or %)."
            ;;
    esac
}

install_desktop() {
    check_exec_path
    locked_setup_venv
    mkdir -p "$(dirname "$DESKTOP_ENTRY")" "$(dirname "$ICON_TARGET")"
    cp "$ICON_SOURCE" "$ICON_TARGET"
    cat > "$DESKTOP_ENTRY" <<EOF
[Desktop Entry]
Type=Application
Version=1.5
Name=$APP_TITLE
GenericName=LoRA Trainer
Comment=Train LoRAs for the Anima image model and preview them
Exec="$LAUNCHER"
Icon=$APP_ID
Terminal=false
Categories=Graphics;
Keywords=LoRA;Anima;training;diffusion;
EOF
    if command -v desktop-file-validate >/dev/null && ! desktop-file-validate "$DESKTOP_ENTRY"; then
        echo "WARN: desktop-file-validate reported the problems above in $DESKTOP_ENTRY;" \
            "the entry is installed anyway." >&2
    fi
    echo "OK: Menu entry installed: $DESKTOP_ENTRY"
}

uninstall_desktop() {
    rm -f "$DESKTOP_ENTRY" "$ICON_TARGET"
    echo "OK: Menu entry removed (the .venv and the data folder are kept)."
}

case "${1:-}" in
    -h | --help) usage; exit 0 ;;
esac
start_log "$@"
case "${1:-}" in
    --install-desktop) install_desktop; exit 0 ;;
    --uninstall-desktop) uninstall_desktop; exit 0 ;;
esac

locked_setup_venv
stop_log
cd "$REPO_DIR"
exec "$APP" "$@"
