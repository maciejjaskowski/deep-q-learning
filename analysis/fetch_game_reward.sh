#!/bin/bash
INSTANCE=`cat ../provision/instance.dns`
scp -i ~/.ssh/gpu-east.pem ubuntu@$INSTANCE:/mnt/dqn/deep-q-learning/log.out .
