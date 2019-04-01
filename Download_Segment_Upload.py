'''
Script to download a file from Amazon S3 to a temporary directory, segment and crop the video there, and pass the results back t
 S3 from whence they came. 
'''
import os
import sys
import boto3 
from boto3.s3.transfer import S3Transfer 
import botocore 
import threading
from Interface_S3 import download, upload 
from Examine_frames import crop_videos_p

## Assume that the user gives as input the Object ID of the video to be processed as a string. 
if __name__ == '__main__':
    vidpath = sys.argv[1]
    configpath = vidpath.split('.avi')[0]+'config.py' 
    ## Download the video to a temp directory: 
    print(vidpath,configpath)
    print('Downloading Video')

    bucket_name = 'froemkelab.videodata'
    download(bucket_name,vidpath)

    print('Downloading Config File')

    download(bucket_name,configpath)


    ## Segment the video:  
    temp_vidpath = '../vmnt/tmp_videos'
    crop_videos_p(temp_vidpath)

    ## Reupload these chunks. 
    keypath_parts = vidpath.split('/')[:-1]
    keypath = os.path.join(*keypath_parts)
    # Find all of the newly produced video files in the temporary directory:
    files = os.listdir(temp_vidpath)
    ident = vidpath.split('/')[-1].split('.')[0]+'cropped_'+'part'
    new_vids = [file for file in files if ident in file]
    print('uploading ',new_vids)
    for filename in new_vids:
        upload(bucket_name,filename,temp_vidpath+'/',keypath)



