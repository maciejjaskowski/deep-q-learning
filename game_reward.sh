#!/bin/bash
scp -i ~/.ssh/gpu-east.pem ubuntu@ec2-54-174-159-174.compute-1.amazonaws.com:/mnt/dqn/deep-q-learning/log.out .
cat log.out | grep "Game reward:" | sed 's/Game reward: \([0-9]*\)/\1/' > rewards.txt
