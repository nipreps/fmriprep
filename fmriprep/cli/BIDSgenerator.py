import os
from subprocess import call
import subprocess
import os.path as op
import shutil
import sys 
import nibabel as nib
import json
import pdb
import numpy as np
from nilearn import image 
import nipype.interfaces.spm as spm

# Create a BIDS directory that contains all your data, renamed according to BIDS structure 
########################
# fromDir = directory where the raw fmri and mri data lives
# toDirRoot = directory where renamed fmri and mri data will live
# Ideally, fromDir and toDirRoot share a root in the project directory 

# subStr = patient identification label (for example, ID, visit, session, all concatinated from 3411 3 2 to 341132)
# toDir = directory in the toDirRoot that is subject-specific with the BIDS format "sub-[ID]"
# Final structure is:

#		Project
#	data	    	      BIDS
#    ID1  ID2  ...      sub-ID1  sub-ID2  ...
#  mri fmri mri fmri   anat func  anat func 
#  .    .    .    .     
#  .    .    .    .

########################
def createBIDS(projDir, pid, visitNum, sessionNum, runName = None):
    fromDir = op.join(projDir,'data', 'imaging', 'participants', pid, 'visit%s' % visitNum,'session%s' % sessionNum)
    toDirRoot =  op.join(projDir, 'data', 'imaging', 'BIDS')             #commented out data
    subStr = '%s%s%s' % (pid,visitNum,sessionNum)
    toDir = op.join(toDirRoot, 'sub-' + subStr)
    if not os.path.exists(toDir):
        os.makedirs(toDir)
        os.makedirs(toDir + '/anat')
        os.makedirs(toDir + '/func')
        print("made new anat and func folder because didn't exist b4")

    # Copy mri images (T1s) to BIDS directory and rename them according to the sub-ID_T1w.nii.gz BIDS format (i.e. sub-341123_T1w.nii.gz)
    ''' Copy over anat '''
    print('Looking for anatomy folder in '+fromDir)
    if os.path.isdir(fromDir + '/anatomical/'):
        print('Anatomy folder found in '+fromDir)
        print(toDir + '/anat/sub-%s_T1w.nii.gz' % subStr)
        if os.path.isfile(fromDir + '/anatomical/T1w.nii.gz'):
             shutil.copyfile(fromDir + '/anatomical/T1w.nii.gz', toDir + '/anat/sub-%s_T1w.nii.gz' % subStr)
        elif os.path.isfile(fromDir + '/anatomical/spgr_defaced.nii.gz'):
             shutil.copyfile(fromDir + '/anatomical/spgr_defaced.nii.gz', toDir + '/anat/sub-%s_T1w.nii.gz' % subStr)
        elif os.path.isfile(fromDir + '/anatomical/spgr_1_defaced.nii.gz'):
             shutil.copyfile(fromDir + '/anatomical/spgr_1_defaced.nii.gz', toDir + '/anat/sub-%s_T1w.nii.gz' % subStr)
        elif os.path.isfile(fromDir + '/anatomical/spgr_2_defaced.nii.gz'):
             shutil.copyfile(fromDir + '/anatomical/spgr_2_defaced.nii.gz', toDir + '/anat/sub-%s_T1w.nii.gz' % subStr)
        elif os.path.isfile(fromDir + '/anatomical/spgr_watershed.nii.gz'):
             shutil.copyfile(fromDir + '/anatomical/spgr_watershed.nii.gz', toDir + '/anat/sub-%s_T1w.nii.gz' % subStr)
        elif os.path.isfile(fromDir + '/anatomical/watershed_spgr.nii.gz'):
             shutil.copyfile(fromDir + '/anatomical/watershed_spgr.nii.gz', toDir + '/anat/sub-%s_T1w.nii.gz' % subStr)
        elif os.path.isfile(fromDir + '/anatomical/watershed_spgr_1.nii.gz'):
             shutil.copyfile(fromDir + '/anatomical/watershed_spgr_1.nii.gz', toDir + '/anat/sub-%s_T1w.nii.gz' % subStr)
        elif os.path.isfile(fromDir + '/anatomical/watershed_spgr_2.nii.gz'):
             shutil.copyfile(fromDir + '/anatomical/watershed_spgr_2.nii.gz', toDir + '/anat/sub-%s_T1w.nii.gz' % subStr)
        elif os.path.isfile(fromDir + '/anatomical/spgr.nii.gz'):
             shutil.copyfile(fromDir + '/anatomical/spgr.nii.gz', toDir + '/anat/sub-%s_T1w.nii.gz' % subStr)
        elif os.path.isfile(fromDir + '/anatomical/wspgr_defaced.nii.gz'):
             shutil.copyfile(fromDir + '/anatomical/wspgr_defaced.nii.gz', toDir + '/anat/sub-%s_T1w.nii.gz' % subStr)

    # Copy fmri data to BIDS directory and rename them according to the sub-[ID]_[task]_bold.nii.gz BIDS format (i.e. sub-341123_restingstate1_bold.nii.gz)
    ''' Copy over func '''
    # Look for functional .nii.gz files
    if runName is None:
        root, tasks, files = next(os.walk(fromDir + '/fmri/'))
        for task in tasks: 
            print(tasks)
            if task == 'preprocessed':
                print("Already preprocessed")
                continue
            if op.exists(os.path.join(root, task)+ '/unnormalized'):
                for subRoot, dirs, files in os.walk(os.path.join(root, task)+ '/unnormalized'):
                    repTime = 0
                    for file in files:
                        if(file.endswith('.json')):
                            with open(os.path.join(subRoot, file)) as f:    
                                data = json.load(f)
                            repTime = data['RepetitionTime']
                            fileName = ('sub-%s_task-' % subStr) + task + '_bold.json'
                            shutil.copyfile(os.path.join(subRoot, file), toDir + '/func/' + fileName)  

                    for file in files:
                        if(file.endswith('.nii.gz')):
                            fileName = ('sub-%s_task-' % subStr) + task + '_bold.nii.gz'
                            if op.exists(os.path.join(subRoot, file)):
                                shutil.copyfile(os.path.join(subRoot, file), toDir + '/func/' + fileName)
                                toFile = toDir + '/func/' + fileName
                                img = nib.load(toFile)
                                hdr = img.get_header()
                                hdr['pixdim'][4] = repTime
                                img.to_filename(img.get_filename())
                                ''' Make sure repetition time was saved to nifti file '''
                                img = nib.load(toFile)
                                #assert(img.get_header()['pixdim'][4] == repTime)     

                # Check if anat and func coordinates are too different, and if they are, fslreorient them
                print("%s/sub-%s/func/sub-%s_task-%s_bold.nii.gz sto_xyz:1"%(toDirRoot,subStr,subStr,task))
                func_coords1 = os.popen("fslval %s/sub-%s/func/sub-%s_task-%s_bold.nii.gz sto_xyz:1"%(toDirRoot,subStr,subStr,task)).read()
                func_coords1 = func_coords1.split()
                func_coords2 = os.popen("fslval %s/sub-%s/func/sub-%s_task-%s_bold.nii.gz sto_xyz:2"%(toDirRoot,subStr,subStr,task)).read()
                func_coords2 = func_coords2.split()
                func_coords3 = os.popen("fslval %s/sub-%s/func/sub-%s_task-%s_bold.nii.gz sto_xyz:3"%(toDirRoot,subStr,subStr,task)).read()
                func_coords3 = func_coords3.split() 
                print(func_coords1)
                coord_vec = [func_coords1[3],func_coords2[3],func_coords3[3]]
                coord_array = np.array([float(coord_vec[0]),float(coord_vec[1]),float(coord_vec[2])])
                brain_vec = np.array([90.000000,-126.000000,-72.000000])
                global_diff = np.subtract(coord_array,brain_vec)
                global_diff = list(global_diff)
                for element in global_diff:
                    if float(element) > 25:
                        print("Reorienting functional, too far from global origin")
                        call(['fslreorient2std %s/sub-%s/func/sub-%s_task-%s_bold.nii.gz %s/sub-%s/func/sub-%s_task-%s_bold_reo.nii.gz' %(toDirRoot,subStr,subStr,task,toDirRoot,subStr,subStr,task)],shell=True)

                        func_reo_coords1 = os.popen("fslval %s/sub-%s/func/sub-%s_task-%s_bold_reo.nii.gz sto_xyz:1"%(toDirRoot,subStr,subStr,task)).read()
                        func_reo_coords1 = func_reo_coords1.split()
                        func_reo_coords2 = os.popen("fslval %s/sub-%s/func/sub-%s_task-%s_bold_reo.nii.gz sto_xyz:2"%(toDirRoot,subStr,subStr,task)).read()
                        func_reo_coords2 = func_reo_coords2.split()
                        func_reo_coords3 = os.popen("fslval %s/sub-%s/func/sub-%s_task-%s_bold_reo.nii.gz sto_xyz:3"%(toDirRoot,subStr,subStr,task)).read()
                        func_reo_coords3 = func_reo_coords3.split() 
                        coord_reo_vec = [func_reo_coords1[3],func_reo_coords2[3],func_reo_coords3[3]]   
                        coord_reo_array = np.array([float(coord_reo_vec[0]),float(coord_reo_vec[1]),float(coord_reo_vec[2])])   
                        global_reo_diff = np.subtract(coord_reo_array,brain_vec)
                        global_reo_diff = list(global_reo_diff)
                        for element in global_reo_diff:
                            if float(element) > 25:
                                print("Not reoriented properly to fit T1 image. Rats. Ask Tianwen what to do.")
                            else:
                                print("Functional close enough to T1, proceed.")
   
                        call(['mv %s/sub-%s/func/sub-%s_task-%s_bold.nii.gz %s/sub-%s/func/sub-%s_task-%s_bold_orig.nii.gz'%(toDirRoot,subStr,subStr,task,toDirRoot,subStr,subStr,task)],shell=True)
                        call(['mv %s/sub-%s/func/sub-%s_task-%s_bold_reo.nii.gz %s/sub-%s/func/sub-%s_task-%s_bold.nii.gz'%(toDirRoot,subStr,subStr,task,toDirRoot,subStr,subStr,task)],shell=True)
                        if os.path.exists('%s/sub-%s/func/sub-%s_task-%s_bold.nii.gz'%(toDirRoot,subStr,subStr,task)) and os.path.exists('%s/sub-%s/func/sub-%s_task-%s_bold_orig.nii.gz'%(toDirRoot,subStr,subStr,task)):
                            call(['/bin/rm -rf %s/sub-%s/func/sub-%s_task-%s_bold_orig.nii.gz'%(toDirRoot,subStr,subStr,task)],shell=True)

            else:
                continue  

    # Copy over the json sidecar files to help with BIDS information
    if runName == None:
        runName = ''
    for root, dirs, files in os.walk(fromDir + '/fmri/' + runName + 'unnormalized'):
        repTime = 0
        for file in files:
            if(file.endswith('.json')):
                with open(os.path.join(root, file)) as f:    
                    data = json.load(f)
                repTime = data['RepetitionTime']
                fileName = ('sub-%s_' % subStr) + runName + '_bold.json'
                shutil.copyfile(os.path.join(root, file), toDir + '/func/' + fileName) 
                print("JSON copied") 

        for file in files:
            if(file.endswith('.nii.gz')):
                fileName = ('sub-%s_' % subStr) + runName + '_bold.nii.gz'
                print(fileName)
                shutil.copyfile(os.path.join(root, file), toDir + '/func/' + fileName)
                toFile = toDir + '/func/' + fileName
                img = nib.load(toFile)
                hdr = img.get_header()
                hdr['pixdim'][4] = repTime
                img.to_filename(img.get_filename())

                ''' Make sure repetition time was saved to nifti file '''
                img = nib.load(toFile)
                #assert(img.get_header()['pixdim'][4] == repTime)

    return toDirRoot, subStr

# Move the preprocessed files generated by the pipeline back to the original rawdata project (project/data) directory in your lab's format
def moveToProject(projDir, pid, visitNum, sessionNum, processedDir, pipeline,runName = None):
    if (runName == None):
        runName = ''
    subStr = '%s%s%s' % (pid,visitNum,sessionNum)
    fromDir = op.join(processedDir, 'fmriprep', 'sub-' + subStr, 'func')
    fromAnatDir = op.join(processedDir,'fmriprep','sub-' + subStr, 'anat')
    toDir = op.join(projDir,'data', 'imaging', 'participants', pid, 'visit%s' % visitNum,'session%s' % sessionNum, 'fmri')
    toAnatDir = op.join(projDir,'data', 'imaging', 'participants', pid, 'visit%s' % visitNum,'session%s' % sessionNum, 'anatomical')

    # Move functional preprocessed (brainmask and preprocessed)
    for root, dirs, files in os.walk(fromDir):
        for file in files:
            if(file.endswith('.nii.gz')):
                start = 'sub-%s_task-' % subStr
                end = '_bold'
                fileName = file.split('.')[0]
                taskName = (file.split(start))[1].split(end)[0]
                toSubDir = op.join(toDir, taskName, pipeline)
                if not op.exists(toSubDir):
                    os.makedirs(toSubDir)
                if 'brainmask' in fileName:
                    shutil.copyfile(op.join(root, file), toSubDir +  '/brainmask.nii.gz')   
                elif 'preproc' in fileName:
                    shutil.copyfile(op.join(root, file), toSubDir + '/I_preproc.nii.gz')

    # Move motion parameters and rename them 
    for root, dirs, files in os.walk(processedDir):
        for file in files:
            if(file.endswith('.par')):
                parts = file.split('-')
                taskName = parts[2][:-20]
                call(['cp '+ op.join(root,file)+ ' ' + op.join(root,'rp_I.txt')],shell=True)
                shutil.copyfile(op.join(root,'rp_I.txt'), toDir + '/' + taskName + '/')

    # Move anatomical processed (brainmask and preprocessed)
    for root, dirs, files in os.walk(fromAnatDir):
        for file in files:
            if(file.endswith('.nii.gz')):
                start = 'sub-%s_T1w_' % subStr
                fileName = file.split('.')[0]
                toSubDir = op.join(toAnatDir, 'preprocessed')
                if not op.exists(toSubDir):
                    os.makedirs(toSubDir)
                if 'brainmask' in fileName:
                    shutil.copyfile(op.join(root, file), toSubDir +  '/T1w_brainmask.nii.gz')
                elif 'preproc' in fileName:
                    shutil.copyfile(op.join(root, file), toSubDir + '/T1w_preproc.nii.gz')

# Function may be removed. Use nilearn to smooth the preprocessed data in the original data folder 
def smooth_preprocessed_data(projDir, pid, visitNum, sessionNum, kernel_width):
    preprocessed_dir = op.join(projDir,'data', 'imaging', 'participants', str(pid), 'visit%s' % str(visitNum),'session%s' % str(sessionNum), 'fmri')
    Dirlist, Flist = [], []
    to_smooth_set = set()
    for dirlist, subdirlist, flist in os.walk(preprocessed_dir):
        Dirlist.append(dirlist)
        Flist.append(flist)
    for Directory in Dirlist:
        for File in Flist:
            for name in File:
                if op.exists(Directory+'/'+name):
                    to_smooth_set.add(Directory+'/'+name)
    smooth_list = list(to_smooth_set)
    kw = int(kernel_width)
    
    for path in smooth_list:
        nilearn_smoothed = image.smooth_img(path,fwhm=kernel_width)
        path_prefix = path.split('.')
        path_prefix = path_prefix[0:-2]
        path_prefix = '.'.join(path_prefix)+'_smoothed_'+str(kernel_width)
        nib.save(nilearn_smoothed,path_prefix)
