from future.utils import raise_from

from nipype import logging

LOGGER = logging.getLogger('workflow')


def is_fmap_type(fmap_type, filename):
    from fmriprep.utils import misc
    import re
    return re.search(misc.fieldmap_suffixes[fmap_type], filename)


def sort_fmaps(fieldmaps):  # i.e. filenames
    from fmriprep.utils import misc
    from fmriprep.workflows.fieldmap.base import is_fmap_type
    fmaps = {}
    for fmap_type in misc.fieldmap_suffixes.keys():
        fmaps[fmap_type] = []
        fmaps[fmap_type] = [doc for doc in fieldmaps
                            if is_fmap_type(fmap_type, doc)]
    return fmaps


def fieldmap_decider(fieldmap_data, settings):
    ''' Run fieldmap_decider to automatically find a
    Fieldmap preprocessing workflow '''

    # POSSIBLE FILES ACCORDING TO BIDS 1.0.0
    # 8.9.1 one phase diff image, at least one magnitude image
    # 8.9.2 two phase images, two magnitude images
    # 8.9.3 fieldmap image (and one magnitude image)
    # 8.9.4 multiple phase encoded directions (topup)

    # inputs = { 'fieldmaps': None }
    # outputs = { 'fieldmaps': None,
    #             'outputnode.mag_brain': None,
    #             'outputnode.fmap_mask': None,
    #             'outputnode.fmap_fieldcoef': None,
    #             'outputnode.fmap_movpar': None}

    try:
        fieldmap_data[0]
    except IndexError as e:
        raise_from(NotImplementedError("No fieldmap data found"), e)

    for filename in subject_data['fieldmaps']:
        if is_fmap_type('phasediff', filename):  # 8.9.1
            from fmriprep.workflows.fieldmap.phase_diff_and_magnitudes import phase_diff_and_magnitudes
            return phase_diff_and_magnitudes(settings)
        elif is_fmap_type('phase', filename):  # 8.9.2
            raise NotImplementedError("No workflow for phase fieldmap data")
        elif is_fmap_type('fieldmap', filename):  # 8.9.3
            from fmriprep.workflows.fieldmap.fieldmap_to_phasediff import fieldmap_to_phasediff
            return fieldmap_to_phasediff(settings=settings)
        elif is_fmap_type('topup', filename):  # 8.0.4
            from fmriprep.workflows.fieldmap.se_fmap_workflow import se_fmap_workflow
            return se_fmap_workflow(settings=settings)

    raise IOError("Unrecognized fieldmap structure")


def create_encoding_file(input_images, in_dict):
    """Creates a valid encoding file for topup"""
    import json
    import nibabel as nb
    import numpy as np
    import os

    if not isinstance(input_images, list):
        input_images = [input_images]
    if not isinstance(in_dict, list):
        in_dict = [in_dict]

    pe_dirs = {'i': 0, 'j': 1, 'k': 2}
    enc_table = []
    for fmap, meta in zip(input_images, in_dict):
        line_values = [0, 0, 0, meta['TotalReadoutTime']]
        line_values[pe_dirs[meta['PhaseEncodingDirection'][0]]] = 1 + (
            -2*(len(meta['PhaseEncodingDirection']) == 2))

        nvols = 1
        if len(nb.load(fmap).shape) > 3:
            nvols = nb.load(fmap).shape[3]

        enc_table += [line_values] * nvols

    np.savetxt(os.path.abspath('parameters.txt'), enc_table, fmt=['%0.1f', '%0.1f', '%0.1f', '%0.20f'])
    return os.path.abspath('parameters.txt')


def mcflirt2topup(in_files, in_mats, out_movpar=None):
    """
    Converts a list of matrices from MCFLIRT to the movpar input
    of TOPUP (a row per file with 6 parameters - 3 translations and 3 rotations
    in this particular order).

    """

    import os.path as op
    import numpy as np
    params = np.zeros((len(in_files), 6))

    if in_mats:
        if len(in_mats) != len(in_files):
            raise RuntimeError('Number of input matrices and files do not match')
        else:
            raise NotImplementedError

    if out_movpar is None:
        fname, fext = op.splitext(op.basename(in_files[0]))
        if fext == '.gz':
            fname, _ = op.splitext(fname)
        out_movpar = op.abspath('./%s_movpar.txt' % fname)

    np.savetxt(out_movpar, params)
    return out_movpar
