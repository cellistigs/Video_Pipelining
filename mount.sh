#!/bin/sh
# Script to mount S3 drive onto this EC2 instance 
echo Mounting Drive
sudo s3fs froemkelab.videodata ~/DeepLabCut/videos/S3 -o use_cache=/tmp -o allow_other -o uid=1000 -o gid=1000 -o mp_umask=227 -o multireq_max=5 -o use_path_request_style -o url=https://s3.us-east-1.amazonaws.com
