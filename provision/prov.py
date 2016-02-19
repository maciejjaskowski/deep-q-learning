import provision
from subprocess import call
import time
import boto3
import sys

ec2 = boto3.client('ec2')

provisionId = sys.argv[1].zfill(10)
print('provisionId: ', provisionId)

instance = provision.provision(provisionId, 'us-east-1a')
instance = provision.provision(provisionId, 'us-east-1a')
print(instance['public_dns_name'])

# print("Wait until running.")
# ec2.wait_until_running(Filters=[{'Name': 'instance-id', 'Values': instance['instance']['InstanceId']}])
#
# print("Running!")

with open('instance.dns', 'w') as f:
  f.write(str(instance['public_dns_name']))

#provision.attach_volume(instance['instance'])

#time.sleep(10000)
#ret = call(["./mount.sh"])
#if ret != 0:
#  print("Problem ", ret)
