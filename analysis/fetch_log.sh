#!/bin/bash
set -e
set -u
echo "Syncing s3://$1/logs with $1/logs."

aws s3 sync s3://$1/logs $1/logs
