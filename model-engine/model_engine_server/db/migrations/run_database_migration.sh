#!/bin/bash

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Change directory to the directory of this script
cd $DIR

# Runs database migration
alembic upgrade head