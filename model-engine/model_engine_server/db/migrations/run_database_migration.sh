#!/bin/bash

# Get the directory of this script without relying on external coreutils.
SCRIPT_PATH="${BASH_SOURCE[0]}"
DIR="$(cd -- "${SCRIPT_PATH%/*}" >/dev/null 2>&1 && pwd)"

# Change directory to the directory of this script
cd "$DIR"

# Runs database migration
alembic upgrade head
