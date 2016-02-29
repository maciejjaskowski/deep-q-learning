#!/bin/bash
set -e
set -u
name=`basename \`pwd\``
echo "Syncing s3://$name/logs with logs."

aws s3 sync s3://"$name"/logs logs

echo "Syncing s3://$name/config with config."
aws s3 sync s3://"$name"/config config

echo "Syncing s3://$name/analysis with analysis."
aws s3 sync s3://"$name"/analysis analysis

