import pytest 
from click.testing import CliRunner
import os
import sys

testdir = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(0,os.path.join(testdir,"../"))
print(os.listdir("."))

from run_dlc import main
def test_run_dlc():
    videodir = "/home/ubuntu/Video_Pipelining/tests/test_mats/temp_videofolder/"
    config = "/home/ubuntu/carceadata_09_09_21-Taiga-2021-09-09_old_data_reformatted/config.yaml"
    runner = CliRunner()
    result =runner.invoke(main,["-d", config, "--v", videodir])
    import pdb; pdb.set_trace()
        
