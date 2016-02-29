#!/bin/bash
set -e
set -u
name=`basename \`analysis\``
echo "Syncing s3://$name/logs with $name/logs."

aws s3 sync s3://"$name"/logs logs

echo "Syncing s3://$name/config with $1/config."
aws s3 sync "s3://$name/config" config

echo "Syncing s3://$name/analysis with $1/analysis."
aws s3 sync "s3://$name/analysis" analysis

