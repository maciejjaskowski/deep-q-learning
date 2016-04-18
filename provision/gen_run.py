import boto3
from subprocess import call
import botocore


def upload_user_data(**kargs):

    s3 = boto3.resource('s3')
    s3.create_bucket(ACL='private', Bucket=kargs['exp_name'])
    k = s3.Object(kargs['exp_name'], 'config/run.sh')

    try:
        script = k.get()['Body'].read()
    except botocore.exceptions.ClientError as e:

        print("No run.sh not found. Uploading.")
        script = """#!/bin/bash
cd /usr/local/cuda/samples/1_Utilities/deviceQuery && make && ./deviceQuery

cd /home/{user_name}

sudo su {user_name} -c "mkdir -p /home/{user_name}/.aws"

sudo su {user_name} -c "aws s3 sync s3://dqn-setup /home/{user_name}/dqn-setup"


sudo su {user_name} -c "git clone https://github.com/maciejjaskowski/{project_name}.git"
sudo su {user_name} -c "cd {project_name} && git reset --hard {sha1}"
sudo su {user_name} -c "mkdir -p /home/{user_name}/{project_name}/weights"
sudo su {user_name} -c "mkdir -p /home/{user_name}/{project_name}/logs"
sudo su {user_name} -c "mkdir -p /home/{user_name}/{project_name}/record0.1"
sudo su {user_name} -c "cp /home/{user_name}/dqn-setup/*.bin /home/{user_name}/{project_name}/"

sudo su ubuntu -c 'aws s3 sync s3://{exp_name}/weights /home/{user_name}/{project_name}/weights'

export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:;
export LD_LIBRARY_PATH=/usr/local/cuda/lib64;
echo $PATH > /home/{user_name}/path.log;
echo $LD_LIBRARY_PATH /home/{user_name}/ld.log;
cd /home/{user_name}/{project_name}
THEANO_FLAGS='floatX=float32,mode=FAST_RUN,allow_gc=False,device=gpu,lib.cnmem=0.9' python run.py --game=space_invaders --dqn.network=nature_with_pad --dqn.updates=deepmind_rmsprop 2>&1 | multilog t s500000 '!tai64nlocal|gzip' ./logs &

watch -n 60 "sudo su {user_name} -c 'aws s3 sync /home/{user_name}/{project_name}/weights s3://{exp_name}/weights' && sudo su {user_name} -c 'aws s3 sync /home/{user_name}/{project_name}/logs s3://{exp_name}/logs' && echo \`date\` >> /home/{user_name}/last_sync" &
        """.format(**kargs)

        k.put(Body=script)
        call(["aws", "s3", "sync", "../analysis", "s3://" + kargs['exp_name'] + "/analysis"])
        call(["mkdir", "-p", "../" + kargs['exp_name']])
        call(["cp", "../analysis/sync.sh", "../" + kargs['exp_name']])
    else:
        print("run.sh found on S3. Reusing.")

    return script
