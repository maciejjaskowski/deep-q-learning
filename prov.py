import provision

instance = provision.provision('000000000005', 'us-east-1a')
instance = provision.provision('000000000005', 'us-east-1a')
with open('ssh.sh', 'w') as f:
  f.write("""#!/bin/bash\necho "sshing to """ + str(instance['public_dns_name']) + """ "\nssh -i  ~/.ssh/gpu-east.pem ubuntu@""" + str(instance['public_dns_name']) + " $1")
provision.attach_volume(instance['instance'])

