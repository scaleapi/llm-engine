#!/bin/bash
# Prepend pip-installed NVIDIA CUDA lib paths so they take priority over stale
# system-level libs (e.g. CUDA 12.9.x in the base image).
if [ -n "$NVIDIA_PIP_LIB_PATH_FILE" ] && [ -f "$NVIDIA_PIP_LIB_PATH_FILE" ]; then
    _nvidia_lib_path="$(cat "$NVIDIA_PIP_LIB_PATH_FILE")"
    if [ -n "$_nvidia_lib_path" ]; then
        export LD_LIBRARY_PATH="${_nvidia_lib_path}:${LD_LIBRARY_PATH:-}"
    fi
fi
exec "$@"
