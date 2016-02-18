#!/bin/bash
set -e
set -u
INSTANCE=`cat instance.dns`
ssh -oStrictHostKeyChecking=no -i  ~/.ssh/gpu-east.pem ubuntu@$INSTANCE 'bash -s' < mount_remote.sh
