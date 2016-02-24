import boto3
import base64
import datetime
import time
from datetime import datetime

ec2 = boto3.client('ec2')


def _prices(az):
    a = ec2.describe_spot_price_history(StartTime=datetime(2016, 1, 1), InstanceTypes=['g2.2xlarge'],
                                        AvailabilityZone=az, ProductDescriptions=['Linux/UNIX'])
    return sorted(a['SpotPriceHistory'], key=lambda s: s['Timestamp'])


def prices():
    return zip(_prices('us-east-1a'), _prices('us-east-1b'), _prices('us-east-1c'), _prices('us-east-1e'))


def provision(client_token, availability_zone):
    user_data = """#!/bin/bash
    cd /usr/local/cuda/samples/1_Utilities/deviceQuery && make && ./deviceQuery

    docker run  --rm -d --name dqn -v /home/ubuntu/dqn:/mnt/dqn $DOCKER_NVIDIA_DEVICES mjaskowski/dqn /mnt/dqn/deep-q-learning/run-gpu
    """

    #dev_sda1 = ec2.blockdevicemapping.EBSBlockDeviceType()
    #dev_sda1.size = 30
    #bdm = ec2.blockdevicemapping.BlockDeviceMapping()
    #bdm['/dev/sda1'] = dev_sda1

#ec2.wait_until_exists
#wait_until_running()
#/dev/sdf    /data   ext3    defaults    1 1
#overlayroot http://stackoverflow.com/questions/19575348/tricks-to-make-an-aws-spot-instance-persistent
    result = ec2.request_spot_instances(DryRun=False,
                                        ClientToken=client_token,
                                        SpotPrice='0.20',
                                        InstanceCount=1,
                                        AvailabilityZoneGroup=availability_zone,
                                        Type='persistent',
                                        LaunchSpecification={
                                            'ImageId': 'ami-60675a0a',
                                            'KeyName': 'gpu-east',
                                            'InstanceType': 'g2.2xlarge',
                                            'Placement': {
                                                'AvailabilityZone': availability_zone
                                            },
                                            'BlockDeviceMappings': [{
                                                'DeviceName': '/dev/sda1',
                                                'Ebs': {
                                                    'VolumeSize': 50,
                                                    'SnapshotId': 'snap-3fb82a22',
                                                    'DeleteOnTermination': False,
                                                    'VolumeType': 'standard',
                                                    'Encrypted': False
                                                }
                                            }],
                                            'EbsOptimized': True,
                                            'Monitoring': {
                                                'Enabled': True
                                            },
                                            'UserData': base64.b64encode(user_data.encode("ascii")).decode('ascii'),
                                            'SecurityGroupIds': ['sg-ab1236d2'],

                                        })

    req_id = result['SpotInstanceRequests'][0]['SpotInstanceRequestId']

    instance_running = False

    instance_description = None
    public_dns_name = None
    while True:
        instance = ec2.describe_spot_instance_requests(SpotInstanceRequestIds=[req_id])['SpotInstanceRequests'][0]
        print(datetime.now().time(), instance['Status']['Message'])

        if 'Fault' in instance.keys():
            return {
                status: instance['Status'],
                instance: instance
            }

        if 'InstanceId' in instance.keys():
            instance_running = True
            instance_description = ec2.describe_instances(InstanceIds=[instance['InstanceId']])
            public_dns_name = instance_description['Reservations'][0]['Instances'][0]['PublicDnsName']
            break
        time.sleep(1)
    # print("Wait until Running!")
    # result.wait_until_running()
    # print("Running!")

    return {
        'status': instance['Status'],
        'instance': instance,
        'instance_description': instance_description,
        'public_dns_name': public_dns_name,
        'run-docker': "ssh -i ~/.ssh/gpu-east.pem ubuntu@" + public_dns_name + " 'bash -s' < ./run_docker.sh"
    }


def attach_volume(instance):
    while True:
        try:
            ec2.attach_volume(
                DryRun=False,
                VolumeId='vol-32b4d7ed',
                InstanceId=instance['InstanceId'],
                Device='/dev/xvdf')
            time.sleep(1)
        except:
            import traceback
            traceback.print_exc()
            print(datetime.now().time(), "Not ready yet.")
            import sys
            print(sys.exc_info()[2])
        else:
            break


#provision('000000000001', 'us-east-1a')


# EBS dodatkowy (na dane): vol-289d50f7

# vpc-e6d76682
# 30GB SSD

# tag: GPU6
