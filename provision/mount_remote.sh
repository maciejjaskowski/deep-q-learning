#!/bin/bash
sudo mkdir -p /mnt/dqn && sudo mount /dev/xvdf /mnt/dqn

cd /mnt/dqn/deep-q-learning && sudo git pull origin aws

sudo docker pull mjaskowski/dqn

cd /usr/local/cuda/samples/1_Utilities/deviceQuery && make && ./deviceQuery
