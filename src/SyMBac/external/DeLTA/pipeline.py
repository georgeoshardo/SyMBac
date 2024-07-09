"""
This script process an input file / folder with data from a multi-position 
microscopy experiment and produces .mat files describing cells lineage, 
fluorescence etc.

The script can be called from the command line:
    python pipeline.py /path/to/file/or/folder /path/to/output/folder
    
Or run from an IDE by changing file paths.

@author: jblugagne
"""

import config as cfg
# Modules:
import time, cv2, os, sys
import numpy as np
from data import postprocess
import utilities as utils


# Load models:
models = utils.loadmodels()

def process_xp(xpfile,
               positions=None,
               use_bioformats=False,
               prototype=None, 
               fileorder='pct',
               filenamesindexing=1,
               resfolder = None,
               rotation_correction = True,
               write_movie = True,
               verbose = 1):
    '''
    Process entire experiment folder.

    Parameters
    ----------
    xpfile : str or tuple/list of str
        Path to experiment file(s) or folder.
    positions : list or None, optional
        List of positions to process. If multiple files, list of lists of 
        positions. If None, all positions are processed.
        The default is None.
    use_bioformats : bool, optional
        Use bio-formats reader for provided files (necessary for .nd2, .oib, 
        .czi...)
        The default is False.
    prototype : str, optional
        File names prototype. See xpreader() in utilities.py.
        The default is None.
    fileorder : str, optional
        File names order. See xpreader() in utilities.py. 
        The default is 'pct'.
    filenamesindexing : int, optional
        File names base indexing. See xpreader() in utilities.py.
        The default is 1.
    resfolder : str, optional
        Path to save results to. If None, a folder within the experiment folder
        will be created.
        The default is None.
    rotation_correction : bool, optional
        Flag to automatically correct the angle of images. 
        The default is True.
    write_movie : bool, optional
        Flag to write movies of results to disk.
        The default is True.
    verbose: int, optional
        Verbosity of console output.
        The default is 1.

    Returns
    -------
    None.

    '''
    
    # If single file path str, turn into list:    
    if type(xpfile) is str:
        xpfile = [xpfile]
    
    # Loop through experiments:
    for x, experiment in enumerate(xpfile):
        
        # Print start message:
        if verbose >= 1:
            print('\n%s ----- Processing experiment %d/%d: %s'%(time.strftime("%e %b %Y - %T", 
                                                                            time.localtime()),
                                                              x+1,
                                                              len(xpfile),
                                                              os.path.split(experiment)[1])
                  )
        
        # Create folders if they don't exist:
        if resfolder is None:
            if os.path.isdir(experiment):
                resfolder = os.path.join(experiment, 'delta_results')
            else:
                resfolder = os.path.splitext(experiment)[0]+'_delta_results'
        if not os.path.exists(resfolder):
            os.mkdir(resfolder)
        
        # Initialize XP reader:
        xp = utils.xpreader(experiment,
                            prototype=prototype,
                            fileorder=fileorder,
                            filenamesindexing=filenamesindexing,
                            use_bioformats=use_bioformats)
        
        # Process positions to run:
        if positions is None:
            positions_torun = range(xp.positions)
        elif len(xpfile)==1:
            positions_torun = positions
        else:
            positions_torun = positions[x]
        if positions_torun is None:
            positions_torun = range(xp.positions)
        
        # Run over all positions:
        for pos in positions_torun:
            
            # Run processing for position:
            delta_output = process_position(xp,
                                            pos,
                                            verbose=verbose)
    
            # Save results to disk:
            utils.legacysave(os.path.join(resfolder,('Position%06d.mat' % (pos+1))), 
                             experiment, 
                             *delta_output,
                             [xp.y,xp.x,xp.channels,xp.timepoints])
            
            # Write results movie:
            if write_movie:
                if verbose>=1:
                    print('Writing mp4 movie file...')
                movie = utils.results_movie(xp,
                                            pos,
                                            *delta_output,
                                            display=False
                                            )
                utils.vidwrite(movie*255,
                               os.path.join(resfolder,('Position%06d.mp4' % (pos+1))),
                               verbose=0
                               )
            
            
            
    # Done, print message:
    if verbose>=1:
        print('\n\n%s ----- %d/%d experiments processed'%(time.strftime("%e %b %Y - %T", 
                                                                        time.localtime()),
                                                          x+1,
                                                          len(xpfile))
              )
        
def process_position(xp,
                     pos,
                     rotation_correction = True,
                     verbose = 1):
    '''
    Apply DeLTA pipeline to a single position in the experiment

    Parameters
    ----------
    xp : xpreader object
        See xpreader() in utilities.py. 
    pos : int
        Position to process.
    rotation_correction : bool, optional
        Flag to automatically correct the angle of images. 
        The default is True.
    verbose: int, optional
        Verbosity of console output.
        The default is 1.

    Returns
    -------
    res : list
        List of dictionaries containing the lineage and labels stacks.
    chamberboxes : list
        List of dictionaries containing the positions of chamber crop boxes in
        the image.
    rot_corr : float
        Rotation applied to images to get the chambers horizontal.
    drift_values : list
        XY drift values computed by preprocessing step.

    '''
    
    ##### Preprocessing - Rotation correction, Chambers identificaton, Drift:
    start_time = time.time()
    if verbose: print('Pos. %d/%d: Preprocessing...' %(pos+1,xp.positions), end='')
    transframes, fluoframes, chamberboxes, rot_corr, driftvalues = preprocessing(xp, pos)
    if chamberboxes is None:
        if verbose: print('\b\b\b - No chambers detected! Aborting...')
        return None, None, rot_corr, None
    
    
    ##### Cell segmentation:
    if verbose: print('\b\b\b - Segmentation...', end='')
    seg, seg_inputs = segmentation(transframes, chamberboxes)
    
    
    ##### Tracking:
    res = trackinglineagefeatures(transframes, 
                                  fluoframes, 
                                  seg, 
                                  seg_inputs,
                                  chamberboxes)
    
    if verbose: print(' - Done! ['+str(round(time.time() - start_time))+' secs]')
    
    return res, chamberboxes, rot_corr, driftvalues

def preprocessing(xp, pos, rotation_correction = True):
    '''
    Rotation correction, drift correction, loading images...

    Parameters
    ----------
    xp : xpreader object
        See xpreader() in utilities.py. 
    pos : int
        Position to process.
    rotation_correction : bool, optional
        Flag to automatically correct the angle of images. 
        The default is True.

    Returns
    -------
    transframes : 3D array of floats
        Transmitted light images, rescaled to [0, 1] dynamic range. Dimensions
        are timepoints -by- image_size_x -by- image_size_y.
    fluoframes : 4D array of floats
        Transmitted light images, rescaled to [0, 1] dynamic range. Dimensions
        are timepoints -by- fluo_channels -by- image_size_x -by- image_size_y.
    chamberboxes : list
        List of dictionaries containing the positions of chamber crop boxes in
        the image.
    rot_corr : float
        Rotation applied to images to get the chambers horizontal.
    drift_values : list
        XY drift values computed by preprocessing step.

    '''
    
    firstframe = xp.getframes(positions=pos,channels=0,frames=0,rescale=(0,1)) # Get first frame for rotation estimation (We assume first channel is trans)
    # Rotation correction:
    if rotation_correction:
        rot_corr = utils.deskew(firstframe) # Estimate rotation angle
    else:
        rot_corr = 0
    # Get trans frames:
    transframes = xp.getframes(positions=pos,channels=0,rescale=(0,1),rotate=rot_corr) # Now load all trans frames, with rotation correction
    firstframe = np.expand_dims(np.expand_dims(cv2.resize(transframes[0],(512,512)),axis=0),axis=3) # using expand_dims to get it into a shape that the chambers id unet accepts
    # Find chambers, filter results, get bounding boxes:
    chambermask = models['chambers'].predict(firstframe,verbose=0)
    chambermask = cv2.resize(np.squeeze(chambermask),transframes.shape[3:0:-1]) # scaling back to original size
    chambermask = postprocess(chambermask,min_size=cfg.min_chamber_area) # Binarization, cleaning and area filtering
    chamberboxes = utils.getChamberBoxes(np.squeeze(chambermask))
    # If no chambers detected:
    if len(chamberboxes)==0:
        return None, None, None, rot_corr, None
        
    # Drift correction:
    drifttemplate = utils.getDriftTemplate(
        chamberboxes, transframes[0], whole_frame=cfg.whole_frame_drift
        ) # This template will be used as reference for drift correction
    driftcorbox = dict(
        xtl = 0,
        xbr = None,
        ytl = 0,
        ybr = None if cfg.whole_frame_drift else max(chamberboxes,
                                                     key=lambda elem: elem['ytl']
                                                     )['ytl']
        ) # Box to match template
    transframes, driftvalues = utils.driftcorr(transframes, template=drifttemplate, box=driftcorbox) # Run drift corr    
    # Load up fluoresence images, apply drift correction and rotation:
    fluoframes = xp.getframes(positions=pos, channels=[x for x in range(1,xp.channels)], squeeze_dimensions=False, rotate=rot_corr)
    if fluoframes is not None:
        fluoframes = fluoframes[0] # Squeezing out the position dimension
        for f in range(fluoframes.shape[1]): #Go over all channels:
            fluoframes[:,f,:,:], _ = utils.driftcorr(fluoframes[:,f,:,:], drift=driftvalues) # Apply drift correction
    
    return transframes, fluoframes, chamberboxes, rot_corr, driftvalues

def segmentation(transframes, chamberboxes):
    '''
    

    Parameters
    ----------
    transframes : 3D array of floats
        Transmitted light images, rescaled to [0, 1] dynamic range. Dimensions
        are timepoints -by- image_size_x -by- image_size_y.
    chamberboxes : list
        List of dictionaries containing the positions of chamber crop boxes in
        the image.

    Returns
    -------
    seg : 3D array of floats
        Segmentation Unet outputs. Dimensions are chambers*timepoints -by- 
        256 -by- 32.
    seg_inputs : 3D array of floats
        Segmentation Unet inputs. Dimensions are chambers*timepoints -by- 
        256 -by- 32.

    '''
    
    seg_inputs = []
    # Compile segmentation inputs:
    for m, chamberbox in enumerate(chamberboxes):
        for i in range(transframes.shape[0]):
            seg_inputs.append(cv2.resize(utils.rangescale(utils.cropbox(transframes[i],chamberbox),(0,1)),(32,256)))
    seg_inputs = np.expand_dims(np.array(seg_inputs),axis=3) # Format into 4D tensor
    # Run segmentation U-Net:
    seg = models['segmentation'].predict(seg_inputs,verbose=0)
    seg = postprocess(seg[:,:,:,0], min_size=cfg.min_cell_area)
    
    return seg, seg_inputs

def trackinglineagefeatures(transframes, 
                        fluoframes, 
                        seg, 
                        seg_inputs, 
                        chamberboxes,
                        verbose = 1):
    '''
    For each chamber, perform tracking, lineage reconstruction, and feature
    extraction.

    Parameters
    ----------
    transframes : 3D array of floats
        Transmitted light images, rescaled to [0, 1] dynamic range. Dimensions
        are timepoints -by- image_size_x -by- image_size_y.
    fluoframes : 4D array of floats
        Transmitted light images, rescaled to [0, 1] dynamic range. Dimensions
        are timepoints -by- fluo_channels -by- image_size_x -by- image_size_y.
    seg : 3D array of floats
        Segmentation Unet outputs. Dimensions are chambers*timepoints -by- 
        256 -by- 32.
    seg_inputs : 3D array of floats
        Segmentation Unet inputs. Dimensions are chambers*timepoints -by- 
        256 -by- 32.
    chamberboxes : list
        List of dictionaries containing the positions of chamber crop boxes in
        the image.
    verbose: int, optional
        Verbosity of console output.
        The default is 1.

    Returns
    -------
    res : list
        List of dictionaries containing the lineage and labels stacks.

    '''
    
    if verbose: 
        print('\b\b\b - Tracking & Feature extraction ', end='')
        message = ''
    res = []
    for m, chamberbox in enumerate(chamberboxes):
        if verbose: 
            print('\b'*len(message),end='')
            message = 'chamber '+str(m+1)+'/'+str(len(chamberboxes))
            print(message,end='')
        
        
        #### Tracking Unet:
        track, track_inputs, frame_numbers = tracking(seg, 
                                                      seg_inputs, 
                                                      m, 
                                                      transframes.shape[0])
        
        ##### Compile lineage and label stack:
        lin, label_stack = lineage(seg, 
                                   track, 
                                   track_inputs, 
                                   frame_numbers, 
                                   m, 
                                   transframes.shape[0])
            
        ##### Extract features:
        lin, label_stack_resized = features(lin,
                                            label_stack,
                                            fluoframes,
                                            chamberbox,
                                            transframes.shape[0])
        
            
        # Append to res dictionary:
        res.append(dict(lineage=lin,labelsstack=label_stack,labelsstack_resized=label_stack_resized))
    
    return res
    
def tracking(seg, seg_inputs, chamber_number, timepoints):
    '''
    Perform tracking for a single chamber.

    Parameters
    ----------
    seg : 3D array of floats
        Segmentation Unet outputs. Dimensions are chambers*timepoints -by- 
        256 -by- 32.
    seg_inputs : 3D array of floats
        Segmentation Unet inputs. Dimensions are chambers*timepoints -by- 
        256 -by- 32.
    chamber_number : int
        Number of the current chamber.
    timepoints : int
        Number of timepoints in movie.

    Returns
    -------
    track : 4D array of floats.
        Tracking Unet outputs. Dimensions are tracking events -by- 256 -by- 32
        -by- 3.
    track_inputs : 4D array of floats.
        Tracking Unet outputs. Dimensions are tracking events -by- 256 -by- 32
        -by- 4.
    frame_numbers : list
        Frame number / timepoint of each tracking event.

    '''
    
    track_inputs = [] # Here we work chamber by chamber for memory reasons
    frame_numbers = []# To reconstruct
    # Compile tracking inputs:
    for i in range(1,timepoints):
        singleseg = utils.getSinglecells(seg[chamber_number*timepoints + i-1]) # Get single cell masks in the 'old' frame
        for c in range(singleseg.shape[0]):
            frame_numbers.append(i) # For lineage reconstruction later
            track_inputs.append(np.stack((
                    seg_inputs[chamber_number*timepoints + i,:,:,0], # Current trans image
                    singleseg[c], # Mask of one previous cell
                    seg_inputs[chamber_number*timepoints + i-1,:,:,0], # Previous trans image
                    seg[chamber_number*timepoints + i]), # Mask of all current cells
                    axis=-1
                ))
    track_inputs = np.array(track_inputs) # Format into 4D tensor
    # Run U-Net model for tracking:
    if track_inputs.size!=0: # If not empty
        track = models['tracking'].predict(track_inputs,verbose=0)
    else:
        track = None
    
    return track, track_inputs, frame_numbers

def lineage(seg, track, track_inputs, frame_numbers, chamber_number, timepoints):
    '''
    Assemble lineage from tracking outputs.

    Parameters
    ----------
    seg : 3D array of floats
        Segmentation Unet outputs. Dimensions are chambers*timepoints -by- 
        256 -by- 32.
    track : 4D array of floats.
        Tracking Unet outputs. Dimensions are tracking events -by- 256 -by- 32
        -by- 3.
    track_inputs : 4D array of floats.
        Tracking Unet outputs. Dimensions are tracking events -by- 256 -by- 32
        -by- 4.
    frame_numbers : list
        Frame number / timepoint of each tracking event.
    chamber_number : int
        Number of the current chamber.
    timepoints : int
        Number of timepoints in movie.

    Returns
    -------
    lin : list
        List of dictionaries containing the information relative to each cell
        in a particular chamber.
    label_stack : 3D array of uint16
        Images stack labelled with each cell in the chamber, over time. For 
        each frame in the stack, i-valued pixels correspond to the (i-1)-th 
        list element/cell in the lin output (0-based list indexing). The 
        dimensions are timepoints -by- 256 -by- 32.

    '''
    
    label_stack = np.zeros([timepoints,seg.shape[1],seg.shape[2]],dtype=np.uint16)
    lin, label_stack = utils.updatelineage(seg[chamber_number*timepoints], label_stack) # Initialize lineage and label stack on first frame
    for i in range(1,timepoints):
        frame_idxs = [x for x, fn in enumerate(frame_numbers) if fn==i]
        if frame_idxs:
            scores = utils.getTrackingScores(track_inputs[frame_idxs[0],:,:,3], track[frame_idxs])
            attrib = utils.getAttributions(scores)
        else:
            attrib = []
        lin, label_stack = utils.updatelineage(seg[chamber_number*timepoints + i], label_stack, framenb=i, lineage=lin, attrib=attrib) # Because we use uint16, we can only track up to 65535 cells per chamber
    
    return lin, label_stack

def features(lin, label_stack, fluoframes, chamberbox, timepoints):
    '''
    Extract cell features such as fluorescence, cell length etc...

    Parameters
    ----------
    lin : list
        List of dictionaries containing the information relative to each cell
        in a particular chamber.
    label_stack : 3D array of uint16
        Images stack labelled with each cell in the chamber, over time. For 
        each frame in the stack, i-valued pixels correspond to the (i-1)-th 
        list element/cell in the lin output (0-based list indexing).
    fluoframes : 4D array of floats
        Transmitted light images, rescaled to [0, 1] dynamic range. Dimensions
        are timepoints -by- fluo_channels -by- image_size_x -by- image_size_y.
    chamberboxes : list
        Dictionary containing the position of the chamber crop box in the image
        for the current chamber.
    timepoints : int
        Number of timepoints in movie.

    Returns
    -------
    lin : list
        List of dictionaries containing the information relative to each cell
        in a particular chamber (updated with single-cell features).
    label_stack_resized : 3D array of uint16.
        Resized version of label_stack, scaled back to the original chamber 
        images crop box size.

    '''
    
    # Add dictionary keys for the features:
    for l in range(len(lin)):
        lin[l] = {**lin[l], **dict(length=[],width=[],area=[],pixels=[])}
        if fluoframes is not None:
            for f in range(fluoframes.shape[1]): #Go over all channels:
                lin[l] = {**lin[l], **{'fluo'+str(f+1):[]}}
    # Generate new label stack:
    label_stack_resized = np.empty([timepoints,chamberbox['ybr']-chamberbox['ytl'],chamberbox['xbr']-chamberbox['xtl']],dtype=np.uint16)
    for i in range(timepoints): # Go over frames
        # Resize from original stack:
        label_stack_resized[i] = cv2.resize(label_stack[i],label_stack_resized.shape[2:0:-1],interpolation=cv2.INTER_NEAREST) # The stack needs to be resized to the actual chamber box size
        # Get cropped out fluorescence images:
        if fluoframes is not None:
            chamberfluo = np.empty((fluoframes.shape[1],)+label_stack_resized.shape[1:])
            for f in range(fluoframes.shape[1]):
                chamberfluo[f] = utils.cropbox(fluoframes[i,f],chamberbox)
        # Get contours of all cells in frame:
        cells, contours = utils.getcellsinframe(label_stack_resized[i],get_contours=True)
        for c, cellnb in enumerate(cells): # Run through cells in frame
            # Length, width, area:
            rotrect = cv2.minAreaRect(contours[c])
            lin[cellnb]['length'].append(max(rotrect[1]))
            lin[cellnb]['width'].append(min(rotrect[1]))
            lin[cellnb]['area'].append(cv2.contourArea(contours[c]))
            # Pixels list and Fluorescence:
            cellpixels = np.where(label_stack_resized[i]==cellnb+1)
            lin[cellnb]['pixels'].append(np.ravel_multi_index(cellpixels,label_stack_resized.shape[1:]).astype(np.float32)) # Using floats for compatibility with previous version of the pipeline
            if fluoframes is not None:
                for f in range(fluoframes.shape[1]): #Go over all channels:
                    lin[cellnb]['fluo'+str(f+1)].append(np.mean(chamberfluo[f,cellpixels[0],cellpixels[1]]))
            
        # Check if some cells got shrunk out of existence after resize:
        cells_orig = utils.getcellsinframe(label_stack[i],get_contours=False)
        missing_cells = set(cells_orig)-set(cells)
        for cellnb in missing_cells:
            lin[cellnb]['length'].append(np.nan)
            lin[cellnb]['width'].append(np.nan)
            lin[cellnb]['area'].append(np.nan)
            lin[cellnb]['pixels'].append([])
            if fluoframes is not None:
                for f in range(fluoframes.shape[1]): #Go over all channels:
                    lin[cellnb]['fluo'+str(f+1)].append(np.nan)
            
            
                        
    return lin, label_stack_resized

### Main
if __name__=='__main__':
    
    # Default parameters:
    resfolder = None
    bioformats = False
    rotation_correction = cfg.rotation_correction
    prototype = None
    filenamesindexing = 1
    fileorder = 'pct'
    verbose = 1
    
    if len(sys.argv) >= 3: # If command line arguments were passed
        xpfolder = sys.argv[1]
        resfolder = sys.argv[2]
        i = 3
        while i<len(sys.argv):
            if sys.argv[i]=='--bio-formats':
                bioformats = bool(int(sys.argv[i+1]))
                i+=2
            if sys.argv[i]=='--order':
                fileorder = sys.argv[i+1]
                i+=2
            if sys.argv[i]=='--index':
                filenamesindexing = int(sys.argv[i+1])
                i+=2
            if sys.argv[i]=='--proto':
                prototype = sys.argv[i+1]
                i+=2
            if sys.argv[i]=='--rot':
                rotation_correction = float(sys.argv[i+1])
                i+=2
            if sys.argv[i]=='--verbose':
                verbose = int(sys.argv[i+1])
                i+=2
    
    else: # Interactive session:
        
        # Get xp settings:
        print('Experiment type?\n1 - Bio-Formats compatible (.nd2, .oib, .czi, .ome.tif...)\n2 - bioformats2sequence (folder)\n3 - micromanager (folder)\n4 - high-throughput (folder)\n0 - other (folder)\nEnter a number: ',end='')
        answer = int(input())
        print()
        
        # If bioformats file(s):
        if answer is None or answer == 1:
            print('Please select experiment file(s)...')
            xpfile = utils.getxppathdialog(ask_folder=False)
            bioformats = True
        # If folder: 
        else:
            print('Please select experiment folder...')
            xpfile = utils.getxppathdialog(ask_folder=True)
            bioformats = False
            if answer is None or answer == 2:
                prototype = None
                fileorder = 'pct'
                filenamesindexing=1
            elif answer == 3:
                prototype = 'Pos%01d/img_channel%03d_position%03d_time%09d_z000.tif'
                fileorder = 'pcpt'
                filenamesindexing=0
            elif answer == 4:
                prototype = 'chan%02d_img/Position%06d_Frame%06d.tif'
                fileorder = 'cpt'
                filenamesindexing=1
            elif answer == 0:
                print('Enter files prototype: ', end='')
                prototype = input()
                print()
                print('Enter files order: ', end='')
                fileorder = input()
                print()
                print('Enter files indexing: ', end='')
                filenamesindexing = int(input())
                print()
            else:
                raise ValueError('Invalid experiment type')
            print()
    
    # Run:
    process_xp(xpfile,
               use_bioformats=bioformats,
               prototype=prototype,
               fileorder=fileorder,
               filenamesindexing=filenamesindexing,
               resfolder = resfolder,
               rotation_correction = rotation_correction,
               write_movie = cfg.write_mp4_movies,
               verbose = verbose)