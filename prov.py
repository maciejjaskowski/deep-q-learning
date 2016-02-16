import provision

instance = provision.provision('000000000005', 'us-east-1a')
instance = provision.provision('000000000005', 'us-east-1a')
provision.attach_volume(instance['instance'])

print(instance['public_dns_name'])
