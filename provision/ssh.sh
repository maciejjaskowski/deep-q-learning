#!/bin/bash
INSTANCE=`cat instance.dns`
echo "sshing to $INSTANCE"
ssh -i  ~/.ssh/gpu-east.pem ubuntu@$INSTANCE $1
