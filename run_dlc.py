import click
import os
import sys
if sys.platform == 'darwin':
    import wx
    if int(wx.__version__[0]) > 3:
        wx.Thread_IsMain = wx.IsMainThread

os.environ["DLClight"] = "True"
os.environ["Colab"] = "True"
import deeplabcut 

@click.command(help = "run deeplabcut analysis with a given model on data")
@click.option("--dlcconfig","-d",help = "full string path to dlc config file for the model you want to use.")
@click.option("--videodir","-v",help = "full string path to videos you want to analyze with this model. ")
def main(dlcconfig,videodir):
    videos = [os.path.join(videodir,vi) for vi in os.listdir(videodir)]
    deeplabcut.analyze_videos(dlcconfig,videos)

if __name__ == "__main__":
    main()
