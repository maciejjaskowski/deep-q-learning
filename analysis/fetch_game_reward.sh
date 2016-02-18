#!/bin/bash
INSTANCE=`cat ../provision/instance.dns`
scp -i ~/.ssh/gpu-east.pem ubuntu@$INSTANCE:/mnt/dqn/deep-q-learning/log.out .
cat log.out | grep "Game reward:" | sed 's/Game reward: \([0-9]*\)/\1/' > rewards.txt
