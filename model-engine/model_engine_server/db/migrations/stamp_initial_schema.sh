#!/bin/bash

# Usage
# ML_INFRA_DATABASE_URL="postgresql://postgres:password@localhost:54320/postgres" bash stamp_initial_schema.sh

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Change directory to the directory of this script
cd $DIR

# Stamps initial revision to new table
alembic stamp fa3267c80731