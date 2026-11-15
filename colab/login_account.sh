#!/usr/bin/env bash
# One-time login of an additional Google account for the Colab CLI (run inside WSL).
#   wsl -d Ubuntu-24.04 -- bash colab/login_account.sh pro2
# Run that from the project root in PowerShell (currently D:\polyu_class\新建文件夹): WSL starts
# in the current Windows folder, so the relative path works wherever the project lives.
# The CLI prints a URL: open it, sign in with THAT account, paste the code back here.
# Its token/sessions are kept apart under ~/.colab-accounts/<name>; use it with
#   .venv\Scripts\python.exe colab\remote.py run -a <name> -- <script> ...
set -euo pipefail
NAME="${1:?usage: login_account.sh <account-name>}"
C="$HOME/.local/bin/colab"
A="$HOME/.colab-accounts/$NAME"
mkdir -p "$A"
HOME="$A" "$C" usage
HOME="$A" "$C" whoami | grep -i email || true
