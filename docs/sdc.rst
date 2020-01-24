.. _sdc:

Susceptibility Distortion Correction (SDC)
------------------------------------------

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


where :math:`\gamma` is the gyromagnetic ratio. Therefore, the
displacements map :math:`d_\text{PE}(x, y, z)` can be estimated
either via estimating the inhomogeneity map :math:`\Delta B_0(x, y, z)`
(:ref:`sdc_phasediff` and :ref:`sdc_direct_b0`) or
via image registration (:ref:`sdc_pepolar`, :ref:`sdc_fieldmapless`).


Correction methods
~~~~~~~~~~~~~~~~~~

The are five broad families of methodologies for mapping the field:

  1. :ref:`sdc_pepolar` (also called **blip-up/blip-down**):
     acquire at least two images with varying :abbr:`PE (phase-encoding)` directions.
     Hence, the realization of distortion is different between the different
     acquisitions. The displacements map :math:`d_\text{PE}(x, y, z)` is
     estimated with an image registration process between the different
     :abbr:`PE (phase-encoding)` acquisitions, regularized by the
     readout time :math:`T_\text{ro}`.
     Corresponds to 8.9.4 of BIDS.

  2. :ref:`sdc_direct_b0`: some sequences (such as :abbr:`SE (spiral echo)`)
     are able to measure the fieldmap :math:`\Delta B_0(x, y, z)` directly.
     Corresponds to section 8.9.3 of BIDS.

  3. :ref:`sdc_phasediff`: to estimate the fieldmap :math:`\Delta B_0(x, y, z)`,
     these methods   measure the phase evolution in time between two close
     :abbr:`GRE (Gradient Recall Echo)` acquisitions. Corresponds to the sections
     8.9.1 and 8.9.2 of the BIDS specification.

  4. :ref:`sdc_fieldmapless`: FMRIPREP now experimentally supports displacement
     field estimation in the absence of fieldmaps via nonlinear registration.

  5. **Point-spread function acquisition**: Not supported by FMRIPREP.


In order to select the appropriate estimation workflow, the input BIDS dataset is
first queried to find the available field-mapping techniques (see :ref:`sdc_base`).
Once the field-map (or the corresponding displacement field) is estimated, the
distortion can be accounted for (see :ref:`sdc_unwarp`).



Calculating the effective echo-spacing and total-readout time
.............................................................

To solve :ref:`(1) <eq_fieldmap>`, all methods (with the exception of the
fieldmap-less approach) will require information about the in-plane
speed of the :abbr:`EPI (echo-planar imaging)` scheme used in
acquisition by reading either the :math:`T_\text{ro}`
(total-readout time) or :math:`t_\text{ees}` (effective echo-spacing):

.. autofunction:: fmriprep.interfaces.fmap.get_ees
.. autofunction:: fmriprep.interfaces.fmap.get_trt


From the phase-difference map to a field map
............................................

To solve :ref:`(1) <eq_fieldmap>` using a :ref:`phase-difference map <sdc_phasediff>`,
the field map :math:`\Delta B_0(x, y, z)` can be derived from the phase-difference
map:

.. autofunction:: fmriprep.interfaces.fmap.phdiff2fmap


References
..........

.. [Jezzard1995] P. Jezzard, R.S. Balaban
                 Correction for geometric distortion in echo planar images from B0
                 field variations Magn. Reson. Med., 34 (1) (1995), pp. 65-73,
                 doi:`10.1002/mrm.1910340111 <https://doi.org/10.1002/mrm.1910340111>`_.
