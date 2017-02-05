''' Utilities for validating inputs/outputs using nipype's input-modification feature '''
from nipype.utils.filemanip import filename_to_list

def validate(inlist, validator, message=''):
    ''' validator must be a function that returns a boolean
    Usage: wf.connect(in_node, out_node, ((outputs, validate, validator), inputs))
    '''
    if not validator(inlist):
        raise ValueError('Input did not satisfy requirement (' + str(validator) + ').' + message)
    return inlist

def is_nd_nifti(inlist, n_dim):
    ''' for use as the `validator` argument in validate.
    Checks if the input images have the right number of dimensions '''
    import nibabel as nb

    for nifti in filename_to_list(inlist):
        if nb.load(nifti).header['dim'][0] != n_dim:
            return False
    return True

def is_4d_nifti(inlist):
    ''' for use as the `validator` argument in validate.
    Checks if the input images have 4 dimensions '''
    return is_nd_nifti(inlist, 4)

def is_3d_nifti(inlist):
    ''' for use as the `validator` argument in validate.
    Checks if the input images have 3 dimensions '''
    return is_nd_nifti(inlist, 3)
