'''
Script to upload a video to the relevant amazon S3 bucket 
'''
import os
import sys
import boto3 
from boto3.s3.transfer import S3Transfer 
import botocore 
import threading
from Interface_S3 import upload 

if __name__ == "__main__":
    foldername = sys.argv[1]
    bucket_name = sys.argv[2] 
    keypath = sys.argv[3]
    resultpath = sys.argv[4]
    exclude = sys.argv[5]
    
    print(foldername,keypath,bucket_name)
    ## Only reupload analysis results:
    analysis_results = os.listdir(foldername)    
    for filename in analysis_results:
        if filename.split('.')[-1] != exclude:
            ## give the file the right key prefix: 
            key = keypath+'/'+resultpath+'/'+filename 
            print(key,foldername+filename)
            upload(bucket_name,filename,foldername,key)






