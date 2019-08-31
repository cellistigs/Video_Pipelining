## Preprocessing script to be run as part of NCAP workflow for DLC
import sys
from videotools import distribute_render

if __name__ == "__main__":
   dirpath = sys.argv[1]
   configpath = sys.argv[2]
   print('starting to render video')
   distribute_render(configpath,dirpath)
   print('finishing video render')



