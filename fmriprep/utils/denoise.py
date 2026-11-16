# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Configuration and build-time checks for ``svht_denoise`` BOLD denoising.

Adapted from QSIPrep's ``qsiprep.utils.misc`` (``load_svht_config``, ``check_svht_phase``,
``check_svht_extent``) and ``qsiprep.utils.versions.svht_denoise_version``. The DWI-only
shell demeaning is left out.
"""

import json
import math
import re
import shutil
import subprocess
from functools import cache
from pathlib import Path

#: svht_denoise cannot estimate a noise level, and so refuses ``-noise``, from fewer volumes
SVHT_MIN_VOLUMES = 3

#: The largest series svht_denoise accepts (``DN_MAX_VOL`` in its ``src/dn.h``)
SVHT_MAX_VOLUMES = 4096

#: The largest patch side svht_denoise accepts (``DN_MAX_EXTENT`` in its ``src/dn.h``)
SVHT_MAX_EXTENT = 27

#: The smallest patch side svht_denoise accepts (``dn_geom_init`` in its ``src/dn_patch.c``)
SVHT_MIN_EXTENT = 3

_SVHT_ENUM_PARAMETERS = {
    'aggregator': ('exclusive', 'gaussian', 'uniform'),
    'filter_method': ('optshrink', 'optthresh', 'truncate'),
    'shape': ('cube', 'sphere'),
}

#: Keys a ``--bold-denoise-config`` file may set. Any key a file omits is left at
#: svht_denoise's own default.
SVHT_CONFIG_KEYS = frozenset(_SVHT_ENUM_PARAMETERS) | {
    'aggregator_fwhm',
    'extent',
    'noise_dof',
    'preserve_noise_bias',
    'stride',
    'vst',
}

#: The parameters that describe the variance-stabilizing transform
_VST_PARAMETERS = ('noise_dof', 'preserve_noise_bias')


class _DuplicateKeyError(ValueError):
    pass


def _reject_duplicate_keys(pairs):
    keys = [key for key, _ in pairs]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise _DuplicateKeyError(f'repeats key(s) {", ".join(duplicates)}')

    return dict(pairs)


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value):
    if not isinstance(value, int | float) or isinstance(value, bool):
        return False

    try:
        return math.isfinite(value)
    except OverflowError:
        # A JSON integer too large to convert to a float
        return False


def load_svht_config(path):
    """Load and check a ``--bold-denoise-config`` JSON file for svht_denoise.

    Parameters
    ----------
    path : str or os.PathLike
        The configuration file.

    Returns
    -------
    dict
        :class:`~fmriprep.interfaces.denoise.SVHTDenoise` input values, as the file gives
        them.

    Raises
    ------
    ValueError
        If the file does not exist, cannot be read, is not a JSON object, repeats a key,
        has an unknown key or an invalid value, or combines settings that svht_denoise
        refuses.
    """
    source = f'svht_denoise configuration file {path}'
    path = Path(path)
    if not path.exists():
        raise ValueError(f'{source} does not exist.')

    try:
        cfg = json.loads(
            path.read_text(encoding='utf-8'),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except _DuplicateKeyError as err:
        raise ValueError(f'{source} {err}.') from err
    except OSError as err:
        raise ValueError(f'{source} could not be read: {err}') from err
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        raise ValueError(f'{source} is not valid JSON: {err}') from err

    if not isinstance(cfg, dict):
        # A ValueError, like every other problem with the file's contents
        raise ValueError(f'{source} must contain a JSON object.')

    unknown = sorted(set(cfg) - SVHT_CONFIG_KEYS)
    if unknown:
        raise ValueError(
            f'{source} has unknown key(s) {", ".join(unknown)}; '
            f'valid keys are {", ".join(sorted(SVHT_CONFIG_KEYS))}.'
        )

    for name, value in cfg.items():
        if name in _SVHT_ENUM_PARAMETERS:
            choices = _SVHT_ENUM_PARAMETERS[name]
            if not isinstance(value, str) or value not in choices:
                raise ValueError(
                    f'{source} sets {name}={value!r}; must be one of {", ".join(choices)}.'
                )
        elif name in ('preserve_noise_bias', 'vst'):
            if not isinstance(value, bool):
                raise ValueError(f'{source} sets {name}={value!r}; must be true or false.')
        elif name == 'noise_dof':
            if not _is_int(value) or not 1 <= value <= 64:
                raise ValueError(
                    f'{source} sets {name}={value!r}; must be an integer from 1 to 64.'
                )
        elif name == 'stride':
            if not _is_int(value) or value not in (1, 2):
                raise ValueError(f'{source} sets {name}={value!r}; must be 1 or 2.')
        elif name == 'extent':
            if (
                not _is_int(value)
                or value < SVHT_MIN_EXTENT
                or value % 2 == 0
                or value > SVHT_MAX_EXTENT
            ):
                raise ValueError(
                    f'{source} sets {name}={value!r}; must be an odd integer from '
                    f'{SVHT_MIN_EXTENT} to {SVHT_MAX_EXTENT}.'
                )
        elif name == 'aggregator_fwhm':
            if not _is_number(value) or value <= 0:
                raise ValueError(f'{source} sets {name}={value!r}; must be a number > 0.')

    # The combinations svht_denoise refuses before reading any data
    aggregator = cfg.get('aggregator', 'gaussian')
    if 'aggregator_fwhm' in cfg and aggregator != 'gaussian':
        raise ValueError(f'{source} sets "aggregator_fwhm", which needs "aggregator" "gaussian".')

    if cfg.get('stride') == 2 and aggregator == 'exclusive':
        raise ValueError(
            f'{source} sets "stride" 2 with "aggregator" "exclusive", which needs a patch '
            'centered on every voxel.'
        )

    vst_options = sorted(set(_VST_PARAMETERS) & set(cfg))
    if vst_options and cfg.get('vst') is False:
        raise ValueError(
            f'{source} sets {" and ".join(vst_options)}, which describe the '
            'variance-stabilizing transform, with "vst" false.'
        )

    return dict(cfg)


def check_svht_phase(params, use_phase):
    """Reject svht_denoise settings that cannot apply to data rotated onto the real axis.

    With phase data, svht_denoise rotates the complex series onto the real axis, where the
    noise is already Gaussian, so it refuses the variance-stabilizing transform and the
    settings that describe it. Whether phase data are available is known only per run, so
    this is checked while the workflow is built.

    Parameters
    ----------
    params : dict
        SVHTDenoise parameters, as returned by :func:`load_svht_config`.
    use_phase : bool
        Whether phase data are used for the run being denoised.
    """
    if not use_phase:
        return

    vst_options = sorted(set(_VST_PARAMETERS) & set(params))
    if params.get('vst'):
        vst_options.insert(0, 'vst')

    if vst_options:
        raise ValueError(
            f'svht_denoise cannot apply {" and ".join(vst_options)} to phase data, which are '
            'rotated onto the real axis and are already Gaussian. Remove them from '
            '--bold-denoise-config or use --ignore phase.'
        )


def svht_phase_units(metadata):
    """Return the ``phase_units`` to pass svht_denoise for a phase file's sidecar.

    Phase in radians is declared, which is right even when the data do not span a full
    turn. Any other unit (BIDS allows ``arbitrary``) is left to svht_denoise, which maps the
    observed range onto one turn.

    Parameters
    ----------
    metadata : dict
        The phase file's sidecar metadata.

    Returns
    -------
    str or None
        ``'radians'``, or ``None`` to let svht_denoise detect the unit.

    Examples
    --------
    >>> svht_phase_units({'Units': 'rad'})
    'radians'
    >>> svht_phase_units({'Units': 'arbitrary'}) is None
    True
    >>> svht_phase_units({}) is None
    True
    """
    return 'radians' if metadata.get('Units') == 'rad' else None


def can_svht_denoise(n_volumes):
    """Whether a series is long enough for svht_denoise to write a noise map.

    A shorter series is not denoised: it takes the same path as with denoising off.
    """
    return n_volumes >= SVHT_MIN_VOLUMES


def svht_auto_extent(n_volumes):
    """Return svht_denoise's default patch extent: the smallest odd k >= 3 with k**3 > n.

    Mirrors ``dn_auto_extent`` in svht_denoise's ``src/dn_patch.c``.

    Examples
    --------
    >>> svht_auto_extent(10), svht_auto_extent(27), svht_auto_extent(243)
    (3, 5, 7)
    """
    extent = SVHT_MIN_EXTENT
    while extent**3 <= n_volumes:
        extent += 2

    return extent


def check_svht_volumes(params, n_volumes, bold_file=None, spatial_shape=None):
    """Reject a series svht_denoise would refuse, before the workflow runs.

    svht_denoise accepts at most :data:`SVHT_MAX_VOLUMES` volumes, and needs ``extent**3``
    to exceed the number of volumes. A cubic patch must also fit within the image, so its
    extent (set, or svht_denoise's default) may not exceed any spatial dimension. These are
    known only per run, so they are checked while the workflow is built.

    Parameters
    ----------
    params : dict
        SVHTDenoise parameters, as returned by :func:`load_svht_config`.
    n_volumes : int
        The number of volumes in the series being denoised.
    bold_file : str, optional
        The series, named in the error message.
    spatial_shape : tuple of int, optional
        The series' spatial dimensions, to check that a cubic patch fits.
    """
    series = f'{bold_file} has' if bold_file else 'The series has'
    if n_volumes > SVHT_MAX_VOLUMES:
        raise ValueError(
            f'{series} {n_volumes} volumes; svht_denoise accepts at most '
            f'{SVHT_MAX_VOLUMES}. Run without --bold-denoise-method svht.'
        )

    extent = params.get('extent')
    if extent is not None and extent**3 <= n_volumes:
        raise ValueError(
            f'{series} {n_volumes} volumes, but svht_denoise needs "extent" cubed to exceed '
            f'the number of volumes, and the "extent" of {extent} in --bold-denoise-config '
            'is too small.'
        )

    if params.get('shape') == 'cube' and spatial_shape is not None:
        cube = extent if extent is not None else svht_auto_extent(n_volumes)
        if cube > min(spatial_shape):
            dims = 'x'.join(str(dim) for dim in spatial_shape)
            raise ValueError(
                f'{series} spatial dimensions {dims}, too small for the {cube}-voxel cubic '
                'patch svht_denoise would use. Use "shape": "sphere" or a smaller "extent" '
                'in --bold-denoise-config.'
            )


@cache
def svht_denoise_version():
    """Return svht_denoise's version (e.g. ``0.1.20261001``), or ``None`` if unavailable."""
    executable = shutil.which('svht_denoise')
    if executable is None:
        return None

    try:
        proc = subprocess.run(
            [executable, '--version'],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    match = re.search(r'svht_denoise (\S+)', proc.stdout)
    return match.group(1) if match else None


_SVHT_FILTERS = {
    'optshrink': (
        'each component was shrunk by the Frobenius-optimal rule [@gavish2017], using a '
        'noise level estimated with the median estimator of @gavish2014'
    ),
    'optthresh': (
        'components were kept by the optimal hard threshold for an unknown noise level '
        '[@gavish2014]'
    ),
    'truncate': (
        'components were truncated at the edge of the noise bulk, using a noise level '
        'estimated with the median estimator of @gavish2014'
    ),
}

_SVHT_AGGREGATORS = {
    'gaussian': (
        'the patch estimates were averaged where they overlapped, weighted by a Gaussian '
        'function of the distance to each patch center [@manjon2013]'
    ),
    'uniform': 'the patch estimates were averaged where they overlapped [@manjon2013]',
    'exclusive': 'each voxel was reconstructed solely from the patch centered on it',
}


def describe_svht(parameters, use_phase, multiecho=False, version=None):
    """Describe svht_denoise BOLD denoising for the methods boilerplate.

    Adapted from QSIPrep's ``describe_svht``.

    Parameters
    ----------
    parameters : dict
        SVHTDenoise parameters, as returned by :func:`load_svht_config`.
    use_phase : bool
        Whether the magnitude and phase data were rotated onto the real axis.
    multiecho : bool
        Whether the series has more than one echo, each denoised separately.
    version : str, optional
        svht_denoise's version.

    Returns
    -------
    str
        Boilerplate text with inline ``[@citation]`` keys.

    Examples
    --------
    >>> print(describe_svht({}, use_phase=False, version='1.0'))
    Before any other processing, thermal noise was removed from the BOLD data.
    The magnitude data were variance-stabilized ... `svht_denoise` 1.0 [@svht_denoise]. ...
    """
    shape = 'cubic' if parameters.get('shape') == 'cube' else 'spherical'
    extent = parameters.get('extent')
    if extent is None:
        # svht_denoise's default, the smallest odd k with k**3 above the volume count
        size = 'sized automatically from the number of volumes'
    elif shape == 'cubic':
        size = f'of {extent}x{extent}x{extent} voxels'
    else:
        size = f'holding at least {extent**3} voxels'

    filter_desc = _SVHT_FILTERS[parameters.get('filter_method', 'optshrink')]
    aggregator_desc = _SVHT_AGGREGATORS[parameters.get('aggregator', 'gaussian')]
    software = '`svht_denoise`' + (f' {version}' if version else '')
    data = 'each echo' if multiecho else 'the series'

    sentences = ['Before any other processing, thermal noise was removed from the BOLD data.']
    vst = parameters.get('vst', True) and not use_phase
    if use_phase:
        sentences.append(
            f'The magnitude and phase data of {data} were rotated onto the real axis '
            '[@nordic], so that the noise remained zero-mean Gaussian, and then denoised'
        )
    elif vst:
        stabilized = (
            'The magnitude data were variance-stabilized to render the noise approximately '
            'Gaussian [@foi2011]'
        )
        if 'noise_dof' in parameters:
            stabilized += f', assuming {parameters["noise_dof"]} receive channels'

        sentences.append(f'{stabilized}, and then denoised')
    else:
        sentences.append('The magnitude data were denoised')

    sentences[-1] += (
        ' by local principal component analysis with optimal singular value shrinkage, '
        f'as implemented in {software} [@svht_denoise]. '
        f'Overlapping {shape} patches, {size}, were decomposed by singular value '
        f'decomposition, {filter_desc}, and {aggregator_desc}.'
    )
    if multiecho:
        sentences.append('Each echo was denoised separately.')

    if use_phase:
        sentences.append('The absolute value of the denoised data was retained.')
    elif vst and not parameters.get('preserve_noise_bias', False):
        sentences.append(
            'The variance-stabilizing transform was inverted at the exact-unbiased operating '
            'point, removing the noise-floor bias from the denoised magnitude data.'
        )

    # Joined into the paragraph it is added to, like the other workflow descriptions
    return '\n'.join(sentences) + '\n'
