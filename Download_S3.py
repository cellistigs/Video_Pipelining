'''
Script to download a video from the relevant amazon S3 bucket into a temporary diretory. 
'''
import sys
import boto3 
import botocore 

def download(BUCKET_NAME,KEY):

    s3 = boto3.resource('s3')

    try:
        s3.Bucket(BUCKET_NAME).download_file(KEY, '~/tmp/'+KEY)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise

if __name__ == "__main__":
    key = sys.argv[1]
    bucket_name = 'froemkelab.videodata'
    download(bucket_name,key):






