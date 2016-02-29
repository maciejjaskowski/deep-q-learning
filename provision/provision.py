import boto3
import base64
import time
import sys
from datetime import datetime
from subprocess import call

ec2 = boto3.client('ec2')


def _prices(az):
    a = ec2.describe_spot_price_history(StartTime=datetime(2016, 1, 1), InstanceTypes=['g2.2xlarge'],
                                        AvailabilityZone=az, ProductDescriptions=['Linux/UNIX'])
    return sorted(a['SpotPriceHistory'], key=lambda s: s['Timestamp'])


def prices():
    return zip(_prices('us-east-1a'), _prices('us-east-1b'), _prices('us-east-1c'), _prices('us-east-1e'))


def upload_user_data(**kargs):

    script = """#!/bin/bash
cd /usr/local/cuda/samples/1_Utilities/deviceQuery && make && ./deviceQuery

cd /home/{user_name}

sudo su {user_name} -c "mkdir -p /home/{user_name}/.aws"

sudo su {user_name} -c "aws s3 sync s3://dqn-setup /home/{user_name}/dqn-setup"


sudo su {user_name} -c "git clone https://github.com/maciejjaskowski/{project_name}.git"
sudo su {user_name} -c "cd {project_name} && git reset --hard {sha1}"
sudo su {user_name} -c "mkdir -p /home/{user_name}/{project_name}/weights"
sudo su {user_name} -c "mkdir -p /home/{user_name}/{project_name}/logs"
sudo su {user_name} -c "cp /home/{user_name}/dqn-setup/space_invaders.bin /home/{user_name}/{project_name}/"

sudo su {user_name} -c "aws s3 sync s3://{exp_name}/weights /home/{user_name}/{project_name}/weights"


export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:;
export LD_LIBRARY_PATH=/usr/local/cuda/lib64;
echo $PATH > /home/{user_name}/path.log;
echo $LD_LIBRARY_PATH /home/{user_name}/ld.log;
cd /home/{user_name}/{project_name}
THEANO_FLAGS='floatX=float32,mode=FAST_RUN,device=gpu,lib.cnmem=0.9' python ex1.py --dqn.network=cnn_gpu 2>&1 | multilog t s100000 '!tai64nlocal|gzip' ./logs &

watch -n 60 "sudo su {user_name} -c 'aws s3 sync /home/{user_name}/{project_name}/weights s3://{exp_name}/weights' && sudo su {user_name} -c 'aws s3 sync /home/{user_name}/{project_name}/logs s3://{exp_name}/logs' && echo \`date\` >> /home/{user_name}/last_sync" &
        """.format(**kargs)

    s3 = boto3.resource('s3')

    s3.create_bucket(ACL='private', Bucket=kargs['exp_name'])

    k = s3.Object(kargs['exp_name'], 'config/run.sh')
    k.put(Body=script)

    call(["aws", "s3", "sync", "../analysis", "s3://" + kargs['exp_name'] + "/analysis"])
    call(["mkdir", "-p", "../" + kargs['exp_name']])
    call(["cp", "sync.sh", "../" + kargs['exp_name']])

    return script


def provision(client_token, availability_zone, spot_price):

    user_data = """#!/bin/bash
      aws s3 sync s3://{exp_name}/config /home/ubuntu/
      chmod a+x /home/ubuntu/run.sh
      /home/ubuntu/run.sh > /home/ubuntu/run_log.out 2> /home/ubuntu/run_log.err
    """.format(exp_name=client_token)

    print(user_data)

    result = ec2.request_spot_instances(DryRun=False,
                                        ClientToken=client_token,
                                        SpotPrice=spot_price,
                                        InstanceCount=1,
                                        AvailabilityZoneGroup=availability_zone,
                                        Type='persistent',
                                        LaunchSpecification={
                                            'ImageId': 'ami-bdd2efd7',
                                            'KeyName': 'gpu-east',
                                            'InstanceType': 'g2.2xlarge',
                                            'Placement': {
                                                'AvailabilityZone': availability_zone
                                            },
                                            'BlockDeviceMappings': [{
                                                'DeviceName': '/dev/sda1',
                                                'Ebs': {
                                                    'VolumeSize': 25,
                                                    'DeleteOnTermination': True,
                                                    'VolumeType': 'standard',
                                                    'Encrypted': True
                                                }
                                            }],
                                            'IamInstanceProfile': {
                                                'Name': 's3'
                                            },
                                            'EbsOptimized': False,
                                            'Monitoring': {
                                                'Enabled': True
                                            },
                                            'UserData': base64.b64encode(user_data.encode("ascii")).decode('ascii'),
                                            'SecurityGroupIds': ['sg-ab1236d2'],

                                        })

    req_id = result['SpotInstanceRequests'][0]['SpotInstanceRequestId']

    print("")
    instance_description = None
    public_dns_name = ''

    while True:
        instance = ec2.describe_spot_instance_requests(SpotInstanceRequestIds=[req_id])['SpotInstanceRequests'][0]
        sys.stdout.write(str(datetime.now().time()) + " " + instance['Status']['Message'] + '\r')
        sys.stdout.flush()

        if 'Fault' in instance.keys():
            return {
                'status': instance['Status'],
                'instance': instance
            }

        if 'InstanceId' in instance.keys():
            instance_description = ec2.describe_instances(InstanceIds=[instance['InstanceId']])
            public_dns_name = instance_description['Reservations'][0]['Instances'][0]['PublicDnsName']

        if public_dns_name != '':
            break

        time.sleep(1)

    return {
        'status': instance['Status'],
        'instance': instance,
        'instance_description': instance_description,
        'public_dns_name': public_dns_name
    }


# def attach_volume(instance):
#     while True:
#         try:
#             ec2.attach_volume(
#                 DryRun=False,
#                 VolumeId='vol-32b4d7ed',
#                 InstanceId=instance['InstanceId'],
#                 Device='/dev/xvdf')
#             time.sleep(1)
#         except:
#             import traceback
#             traceback.print_exc()
#             print(datetime.now().time(), "Not ready yet.")
#             import sys
#             print(sys.exc_info()[2])
#         else:
#             break


def main(availability_zone, spot_price, client_token):

    project_name = "deep-q-learning"
    from subprocess import Popen,PIPE
    process = Popen(["git", "rev-parse", "HEAD"], shell=False, stdout=PIPE)
    sha1, _ = process.communicate(str.encode("utf-8"))
    sha1 = sha1[:-1]

    user_script = upload_user_data(exp_name=client_token, sha1=sha1, user_name="ubuntu", project_name=project_name)

    print("""
    project_name: {project_name}
    sha1: {sha1}

    availability_zone: {availability_zone}
    spot_price: {spot_price}

    client_token: {client_token}
    user_script:

    {user_script}
    """.format(project_name=project_name, sha1=sha1,
               client_token=client_token, user_script=user_script,
               spot_price=spot_price, availability_zone=availability_zone))

    instance = provision(client_token=client_token, availability_zone=availability_zone,
                         spot_price=spot_price)

    print(instance)
    print("public_dns_name: ", instance['public_dns_name'])

    with open('instance.dns', 'w') as f:
        f.write(str(instance['public_dns_name']))

if __name__ == "__main__":
    import sys
    import getopt
    optlist, args = getopt.getopt(sys.argv[1:], '', [
        'price=',
        'client_token='
        ])

    d = {'availability_zone': 'us-east-1a'}
    for o, a in optlist:
        if o in ("--price",):
            d['spot_price'] = a
        elif o in ("--client_token",):
            d['client_token'] = a
        else:
            assert False, "unhandled option"

    main(**d)



