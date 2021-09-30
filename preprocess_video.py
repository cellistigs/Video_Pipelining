# Preprocessing script to be run as part of NCAP workflow for DLC
import sys
from videotools import distribute_render
"""Script to distribute rendering across multiple machines. 


usage: 

```
python preprocess_video.py "dirpath" "configpath" "outpath"
```

where dirpath is a directory with videos, configpath is a path to a configuration file containing regions of interest, and outpath is the directory we should write outputs to. If outpath is none, will write back to directory. 

"""

if __name__ == "__main__":
   dirpath = sys.argv[1]
   configpath = sys.argv[2]
   outpath = sys.argv[3]
   print('starting to render video')
   distribute_render(configpath,dirpath,outpath)
   print('finishing video render')



