#!/bin/bash
set -e
set -u
echo "Syncing s3://$1/logs with $1/logs."

aws s3 sync s3://$1/logs $1/logs

echo "Syncing s3://$1/config with $1/config."
aws s3 sync s3://$1/config $1/config

echo "Syncing s3://$1/analysis with $1/analysis."
aws s3 sync s3://$1/analysis $1/analysis

