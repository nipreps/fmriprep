#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Handling surfaces
-----------------

"""
import os
import re

import numpy as np
import nibabel as nb

from nipype.interfaces.base import (
    BaseInterfaceInputSpec, TraitedSpec, File, traits, isdefined,
    SimpleInterface
)


class NormalizeSurfInputSpec(BaseInterfaceInputSpec):
    in_file = File(mandatory=True, exists=True, desc='Freesurfer-generated GIFTI file')
    transform_file = File(exists=True, desc='FSL or LTA affine transform file')


class NormalizeSurfOutputSpec(TraitedSpec):
    out_file = File(desc='output file with re-centered GIFTI coordinates')


class NormalizeSurf(SimpleInterface):
    """ Normalizes a FreeSurfer-generated GIFTI image

    FreeSurfer includes an offset to the center of the brain volume that is not
    respected by all software packages.
    Normalization involves adding this offset to the coordinates of all
    vertices, and zeroing out that offset, to ensure consistent behavior
    across software packages.
    In particular, this normalization is consistent with the Human Connectome
    Project pipeline (see `AlgorithmSurfaceApplyAffine`_ and
    `FreeSurfer2CaretConvertAndRegisterNonlinear`_), although the the HCP
    may not zero out the offset.

    GIFTI files with ``midthickness``/``graymid`` in the name are also updated
    to include the following metadata entries::

        {
            AnatomicalStructureSecondary: MidThickness,
            GeometricType: Anatomical
        }

    This interface is intended to be applied uniformly to GIFTI surface files
    generated from the ``?h.white``/``?h.smoothwm`` and ``?h.pial`` surfaces,
    as well as externally-generated ``?h.midthickness``/``?h.graymid`` files.
    In principle, this should apply safely to any other surface, although it is
    less relevant to surfaces that don't describe an anatomical structure.

    .. _AlgorithmSurfaceApplyAffine: https://github.com/Washington-University/workbench\
/blob/1b79e56/src/Algorithms/AlgorithmSurfaceApplyAffine.cxx#L73-L91

    .. _FreeSurfer2CaretConvertAndRegisterNonlinear: https://github.com/Washington-University/\
Pipelines/blob/ae69b9a/PostFreeSurfer/scripts/FreeSurfer2CaretConvertAndRegisterNonlinear.sh\
#L147-154

    """
    input_spec = NormalizeSurfInputSpec
    output_spec = NormalizeSurfOutputSpec

    def _run_interface(self, runtime):
        transform_file = self.inputs.transform_file
        if not isdefined(transform_file):
            transform_file = None
        self._results['out_file'] = normalize_surfs(
            self.inputs.in_file,
            transform_file,
            newpath=runtime.cwd
        )
        return runtime


class GiftiNameSourceInputSpec(BaseInterfaceInputSpec):
    in_file = File(mandatory=True, exists=True, desc='input GIFTI file')
    pattern = traits.Str(mandatory=True,
                         desc='input file name pattern (must capture named group "LR")')
    template = traits.Str(mandatory=True, desc='output file name template')


class GiftiNameSourceOutputSpec(TraitedSpec):
    out_name = traits.Str(desc='(partial) filename formatted according to template')


class GiftiNameSource(SimpleInterface):
    """Construct a new filename based on an input filename, a matching pattern,
    and a related template.

    This interface is intended for use with GIFTI files, to generate names
    conforming to Section 9.0 of the `GIFTI Standard`_.

    Patterns are expected to have named groups, including one named "LR" that
    matches "l" or "r".
    These groups must correspond to named format elements in the template.

    .. testsetup::

    >>> open('lh.pial.gii', 'w').close()
    >>> open('rh.fsaverage.gii', 'w').close()

    .. doctest::

    >>> from fmriprep.interfaces import GiftiNameSource
    >>> surf_namer = GiftiNameSource()
    >>> surf_namer.inputs.pattern = r'(?P<LR>[lr])h.(?P<surf>\w+).gii'
    >>> surf_namer.inputs.template = r'{surf}.{LR}.surf'
    >>> surf_namer.inputs.in_file = 'lh.pial.gii'
    >>> res = surf_namer.run()
    >>> res.outputs.out_name
    'pial.L.surf'

    >>> func_namer = GiftiNameSource()
    >>> func_namer.inputs.pattern = r'(?P<LR>[lr])h.(?P<space>\w+).gii'
    >>> func_namer.inputs.template = r'space-{space}.{LR}.func'
    >>> func_namer.inputs.in_file = 'rh.fsaverage.gii'
    >>> res = func_namer.run()
    >>> res.outputs.out_name
    'space-fsaverage.R.func'

    .. testcleanup::

    >>> import os
    >>> os.unlink('lh.pial.gii')
    >>> os.unlink('rh.fsaverage.gii')

    .. _GIFTI Standard: https://www.nitrc.org/frs/download.php/2871/GIFTI_Surface_Format.pdf
    """
    input_spec = GiftiNameSourceInputSpec
    output_spec = GiftiNameSourceOutputSpec

    def _run_interface(self, runtime):
        in_format = re.compile(self.inputs.pattern)
        in_file = os.path.basename(self.inputs.in_file)
        info = in_format.match(in_file).groupdict()
        info['LR'] = info['LR'].upper()
        filefmt = self.inputs.template
        self._results['out_name'] = filefmt.format(**info)
        return runtime


class GiftiSetAnatomicalStructureInputSpec(BaseInterfaceInputSpec):
    in_file = File(mandatory=True, exists=True,
                   desc='GIFTI file beginning with "lh." or "rh."')


class GiftiSetAnatomicalStructureOutputSpec(TraitedSpec):
    out_file = File(desc='output file with updated AnatomicalStructurePrimary entry')


class GiftiSetAnatomicalStructure(SimpleInterface):
    """Set AnatomicalStructurePrimary attribute of GIFTI image based on
    filename.

    For files that begin with ``lh.`` or ``rh.``, update the metadata to
    include::

        {
            AnatomicalStructurePrimary: (CortexLeft | CortexRight),
        }

    If ``AnatomicalStructurePrimary`` is already set, this function has no
    effect.

    """
    input_spec = GiftiSetAnatomicalStructureInputSpec
    output_spec = GiftiSetAnatomicalStructureOutputSpec

    def _run_interface(self, runtime):
        img = nb.load(self.inputs.in_file)
        if any(nvpair.name == 'AnatomicalStruturePrimary' for nvpair in img.meta.data):
            out_file = self.inputs.in_file
        else:
            fname = os.path.basename(self.inputs.in_file)
            if fname[:3] in ('lh.', 'rh.'):
                asp = 'CortexLeft' if fname[0] == 'l' else 'CortexRight'
            else:
                raise ValueError(
                    "AnatomicalStructurePrimary cannot be derived from filename")
            img.meta.data.insert(0, nb.gifti.GiftiNVPairs('AnatomicalStructurePrimary', asp))
            out_file = os.path.join(runtime.cwd, fname)
            img.to_filename(out_file)
        self._results['out_file'] = out_file
        return runtime


def normalize_surfs(in_file, transform_file, newpath=None):
    """ Re-center GIFTI coordinates to fit align to native T1 space

    For midthickness surfaces, add MidThickness metadata

    Coordinate update based on:
    https://github.com/Washington-University/workbench/blob/1b79e56/src/Algorithms/AlgorithmSurfaceApplyAffine.cxx#L73-L91
    and
    https://github.com/Washington-University/Pipelines/blob/ae69b9a/PostFreeSurfer/scripts/FreeSurfer2CaretConvertAndRegisterNonlinear.sh#L147
    """

    img = nb.load(in_file)
    transform = load_transform(transform_file)
    pointset = img.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0]
    coords = pointset.data.T
    c_ras_keys = ('VolGeomC_R', 'VolGeomC_A', 'VolGeomC_S')
    ras = np.array([[float(pointset.metadata[key])]
                    for key in c_ras_keys])
    ones = np.ones((1, coords.shape[1]), dtype=coords.dtype)
    # Apply C_RAS translation to coordinates, then transform
    pointset.data = transform.dot(np.vstack((coords + ras, ones)))[:3].T.astype(coords.dtype)

    secondary = nb.gifti.GiftiNVPairs('AnatomicalStructureSecondary', 'MidThickness')
    geom_type = nb.gifti.GiftiNVPairs('GeometricType', 'Anatomical')
    has_ass = has_geo = False
    for nvpair in pointset.meta.data:
        # Remove C_RAS translation from metadata to avoid double-dipping in FreeSurfer
        if nvpair.name in c_ras_keys:
            nvpair.value = '0.000000'
        # Check for missing metadata
        elif nvpair.name == secondary.name:
            has_ass = True
        elif nvpair.name == geom_type.name:
            has_geo = True
    fname = os.path.basename(in_file)
    # Update metadata for MidThickness/graymid surfaces
    if 'midthickness' in fname.lower() or 'graymid' in fname.lower():
        if not has_ass:
            pointset.meta.data.insert(1, secondary)
        if not has_geo:
            pointset.meta.data.insert(2, geom_type)

    if newpath is not None:
        newpath = os.getcwd()
    out_file = os.path.join(newpath, fname)
    img.to_filename(out_file)
    return out_file


def load_transform(fname):
    """Load affine transform from file

    Parameters
    ----------
    fname : str or None
        Filename of an LTA or FSL-style MAT transform file.
        If ``None``, return an identity transform

    Returns
    -------
    affine : (4, 4) numpy.ndarray
    """
    if fname is None:
        return np.eye(4)

    if fname.endswith('.mat'):
        return np.loadtxt(fname)
    elif fname.endswith('.lta'):
        with open(fname, 'rb') as fobj:
            for line in fobj:
                if line.startswith(b'1 4 4'):
                    break
            lines = fobj.readlines()[:4]
        return np.genfromtxt(lines)

    raise ValueError("Unknown transform type; pass FSL (.mat) or LTA (.lta)")
