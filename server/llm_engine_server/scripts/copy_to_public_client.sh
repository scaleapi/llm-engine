#!/bin/bash
# Usage: bash build_and_publish_to_codeartifact.sh $PATH_TO_PRIVATE_CLIENT $PATH_TO_PUBLIC_CLIENT

set -e
PRIVATE_CLIENT_ROOT=$1
PUBLIC_CLIENT_ROOT=$2

rm -rf $PUBLIC_CLIENT_ROOT/launch/api_client/*
cp -r $PRIVATE_CLIENT_ROOT/launch_client/* $PUBLIC_CLIENT_ROOT/launch/api_client/

sed -i '' 's/launch_client/launch.api_client/g' $(find $PUBLIC_CLIENT_ROOT/launch/api_client -type f -name '*\.py')
