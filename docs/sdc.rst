.. _sdc:

Susceptibility Distortion Correction (SDC)
------------------------------------------
Please note that all routines for susceptibility-derived distortion correction
have been excised off of *fMRIPrep* for utilization on other projects
(e.g., `dMRIPrep <https://www.nipreps.org/dmriprep>`__).
For more detailed documentation on
:abbr:`SDC (susceptibility-derived distortion correction)`
routines, check on `www.nipreps.org/sdcflows <https://www.nipreps.org/sdcflows>`__.

Introduction
~~~~~~~~~~~~
:abbr:`SDC (susceptibility-derived distortion correction)` methods usually try to
make a good estimate of the field inhomogeneity map.
The inhomogeneity map is directly related to the displacement of
a given pixel :math:`(x, y, z)` along the
:abbr:`PE (phase-encoding)` direction (:math:`d_\text{PE}(x, y, z)`) is
proportional to the slice readout time (:math:`T_\text{ro}`)
and the field inhomogeneity (:math:`\Delta B_0(x, y, z)`)
as follows ([Jezzard1995]_, [Hutton2002]_):

  .. _eq_fieldmap:

  .. math::

      d_\text{PE}(x, y, z) = \gamma \Delta B_0(x, y, z) T_\text{ro} \qquad (1)

where :math:`\gamma` is the gyromagnetic ratio.
Therefore, the displacements map :math:`d_\text{PE}(x, y, z)` can be estimated
either via estimating the inhomogeneity map :math:`\Delta B_0(x, y, z)` or
via image registration (see below).

Correction methods
~~~~~~~~~~~~~~~~~~
The are five broad families of methodologies for mapping the field:

  1. **Phase Encoding POLARity** (*PEPOLAR*; also called *blip-up/blip-down*;
     :py:func:`~sdcflows.workflows.pepolar.init_pepolar_unwarp_wf`):
     acquire at least two images with varying :abbr:`PE (phase-encoding)` directions.
     Hence, the realization of distortion is different between the different
     acquisitions. The displacements map :math:`d_\text{PE}(x, y, z)` is
     estimated with an image registration process between the different
     :abbr:`PE (phase-encoding)` acquisitions, regularized by the
     readout time :math:`T_\text{ro}`.
     Corresponds to 8.9.4 of BIDS.
  2. **Direct B0 mapping sequences** (:py:func:`~sdcflows.workflows.fmap.init_fmap_wf`):
     some sequences (such as :abbr:`SE (spiral echo)`)
     are able to measure the fieldmap :math:`\Delta B_0(x, y, z)` directly.
     Corresponds to section 8.9.3 of BIDS.
  3. **Phase-difference B0 mapping** (:py:func:`~sdcflows.workflows.phdiff.init_phdiff_wf`):
     to estimate the fieldmap :math:`\Delta B_0(x, y, z)`,
     these methods   measure the phase evolution in time between two close
     :abbr:`GRE (Gradient Recall Echo)` acquisitions. Corresponds to the sections
     8.9.1 and 8.9.2 of the BIDS specification.
  4. **"Fieldmap-less" estimation** (experimental; :py:func:`~sdcflows.workflows.syn.init_syn_sdc_wf`):
     *fMRIPrep* now experimentally supports displacement
     field estimation in the absence of fieldmaps via nonlinear registration.
  5. **Point-spread function acquisition**: Not supported by BIDS, and hence *fMRIPrep*.

In order to select the appropriate estimation workflow, the input BIDS dataset is
first queried to find the available field-mapping techniques
(see :py:func:`~sdcflows.workflows.base.init_sdc_estimate_wf`).
Once the field-map (or the corresponding displacement field) is estimated, the
distortion can be accounted for
(see :py:func:`~sdcflows.workflows.unwarp.init_sdc_unwarp_wf`).

Automatic selection of the appropriate susceptibility-distortion correction (SDC) method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If the dataset metadata indicate that more than one field map acquisition is
``IntendedFor`` (see the `BIDS Specification
<https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/01-magnetic-resonance-imaging-data.html#fieldmap-data>`__),
the following priority will be used:

  1. :ref:`sdc_pepolar` (or **blip-up/blip-down**)

  2. :ref:`sdc_direct_b0`

  3. :ref:`sdc_phasediff`

  4. :ref:`sdc_fieldmapless`


Table of behavior (fieldmap use-cases):

=============== =========== ============= ===============
Fieldmaps found ``use_syn`` ``force_syn``     Action
=============== =========== ============= ===============
True            *           True          Fieldmaps + SyN
True            *           False         Fieldmaps
False           *           True          SyN
False           True        False         SyN
False           False       False         HMC only
=============== =========== ============= ===============

Calculating the effective echo-spacing and total-readout time
.............................................................
To solve :ref:`(1) <eq_fieldmap>`, all methods (with the exception of the
fieldmap-less approach) will require information about the in-plane
speed of the :abbr:`EPI (echo-planar imaging)` scheme used in
acquisition by reading either the :math:`T_\text{ro}`
(total-readout time) or :math:`t_\text{ees}` (effective echo-spacing).
See corresponding implementations under *SDCFlows*:

  * :py:func:`~sdcflows.interfaces.fmap.get_ees`
  * :py:func:`~sdcflows.interfaces.fmap.get_trt`

From the phase-difference map to a field map
............................................
To solve :ref:`(1) <eq_fieldmap>` using a :ref:`phase-difference map <sdc_phasediff>`,
the field map :math:`\Delta B_0(x, y, z)` can be derived from the phase-difference
map (:py:func:`~sdcflows.interfaces.fmap.phdiff2fmap`)

References
..........

.. [Jezzard1995] P. Jezzard, R.S. Balaban
                 Correction for geometric distortion in echo planar images from B0
                 field variations Magn. Reson. Med., 34 (1) (1995), pp. 65-73,
                 doi:`10.1002/mrm.1910340111 <https://doi.org/10.1002/mrm.1910340111>`_.
.. [Hutton2002] Hutton et al., Image Distortion Correction in fMRI: A Quantitative
                Evaluation, NeuroImage 16(1):217-240, 2002. doi:`10.1006/nimg.2001.1054
                <https://doi.org/10.1006/nimg.2001.1054>`_.