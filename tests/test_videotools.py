import os 
import sys
import pytest

testdir = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(0,os.path.join(testdir,"../"))
from videotools import distribute_render


def test_distribute_render(tmpdir):
    clips = tmpdir / "clipfolder"
    clips.mkdir()

    configpath =  os.path.join(testdir,"test_mats/temp_configfolder/TempTrial25config.yaml")
    dirpath = os.path.join(testdir,"test_mats/temp_videofolder")
    ## Write to a diff directory
    distribute_render(configpath,dirpath,clips)
    clipnames = ['TempTrial25_shortestroi_1cropped_part0.mp4', 'TempTrial25_shortestroi_0cropped_part0.mp4', 'TempTrial25_shortestroi_2cropped_part0.mp4']

    for c in clipnames:
        assert c in os.listdir(clips) 
        assert c not in os.listdir(dirpath)

    ## write to same directory:
    distribute_render(configpath,dirpath)
    for c in clipnames:
        assert c in os.listdir(dirpath)
        os.remove(os.path.join(dirpath,c))

