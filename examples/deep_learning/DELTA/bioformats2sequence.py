"""
This file can be run in the conda environment described in bioformats_env.yml
It will convert microscopy files like .oif, .nd2, .czi... to an images
sequence on disk that can then be read with the xpreader object in pipeline.py
in the main DeLTA environment (All other files in this repository must be run
in that environment. See delta_env.yml for specs)
Unfortunately bioformats only works in python 2, and our code runs in python 3

It can be run in the command line:
    python bioformats2sequence.py microscopyexperiment.czi /path/to/sequence/folder

Or in an IDE by modifying the paths below

@author: jblugagne
"""

import sys, os
from utilities import xpreader
from cv2 import imwrite

# Files:
if len(sys.argv) == 3: # If arguments were passed
    xpfile = sys.argv[1]
    save_folder = sys.argv[2]
else:
    DeLTA_data = 'C:/DeepLearning/DeLTA_data/'
    xpfile = DeLTA_data + 'mother_machine/evaluation/evaluation_movie.nd2'
    save_folder = DeLTA_data + 'mother_machine/evaluation/sequence/'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

# Initialize XP reader:
xp = xpreader(xpfile,
              use_bioformats=True)

# Get Filename prototype:
filename_proto = 'Position%0'+str(len(str(xp.positions)))+'d_'\
    +'Channel%0'+str(len(str(xp.channels)))+'d_'\
        +'Frame%0'+str(len(str(xp.timepoints)))+'d'+\
            '.tif'

# Run through frames, save to disk:
counter = 0.
for pos in range(xp.positions):
    for cha in range(xp.channels):
        for fra in range(xp.timepoints):
            # Get frame:
            frame = xp.getframes(pos,cha,fra)
            # Generate filename:
            filename = filename_proto % (pos+1,cha+1,fra+1)
            # Save to disk:
            imwrite(os.path.join(save_folder,filename),frame)
            # display progress:
            counter += 1
            print('%s - saved (%03d%% done)' \
                  % (filename,100*counter/(xp.positions*xp.channels*xp.timepoints)))

# Close the reader:
xp.close()