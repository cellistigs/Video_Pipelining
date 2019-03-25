'''
Script to download a video from the relevant amazon S3 bucket into a temporary diretory. 
'''
import sys
import boto3 
from boto3.s3.transfer import S3Transfer 
import botocore 
import threading
## from https://stackoverflow.com/questions/41827963/track-download-progress-of-s3-file-using-boto3-and-callbacks
class ProgressPercentage(object):
    def __init__(self,client,BUCKET,KEY):
        self._filename = KEY
        self._size = client.head_object(Bucket=BUCKET,Key=KEY)['ContentLength']
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self,bytes_amount):
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = round((self._seen_so_far/self._size)*100,2)
            sys.stdout.write(
                        "\r%s  %s / %s  (%.2f%%)" % (
                        self._filename, self._seen_so_far, self._size,
                        percentage))
            sys.stdout.flush()


def download(BUCKET_NAME,KEY):

    s3 = boto3.resource('s3')
    # for the purposes of temporary storage, we only use the last bit of the name as an indentifier: 
    USEKEY = KEY.split('/')[-1]

    try:
        transfer = S3Transfer(boto3.client('s3','us-east-1')) 
        progress = ProgressPercentage(transfer._manager._client,BUCKET_NAME,KEY)
        transfer.download_file(BUCKET_NAME,KEY, '../vmnt/tmp_videos/'+USEKEY,callback = progress)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise

if __name__ == "__main__":
    key = sys.argv[1]
    bucket_name = 'froemkelab.videodata'
    download(bucket_name,key)






