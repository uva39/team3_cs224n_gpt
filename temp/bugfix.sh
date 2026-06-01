git config --global --unset-all credential.helper 2>/dev/null || true
git config --local --unset-all credential.helper 2>/dev/null || true
git config --system --unset-all credential.helper 2>/dev/null || true

unset GIT_ASKPASS
unset SSH_ASKPASS
unset VSCODE_GIT_ASKPASS_NODE
unset VSCODE_GIT_ASKPASS_EXTRA_ARGS
unset VSCODE_GIT_ASKPASS_MAIN
unset VSCODE_GIT_IPC_HANDLE

export GIT_TERMINAL_PROMPT=1

git config --global credential.helper cache
git push origin js