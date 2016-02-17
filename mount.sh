#!/bin/bash
set -e
set -u
ssh -oStrictHostKeyChecking=no -i  ~/.ssh/gpu-east.pem ubuntu@$1 'bash -s' < mount_remote.sh
