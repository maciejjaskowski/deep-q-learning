#!/bin/bash
echo "sshing to ec2-54-174-159-174.compute-1.amazonaws.com "
ssh -i  ~/.ssh/gpu-east.pem ubuntu@ec2-54-174-159-174.compute-1.amazonaws.com $1