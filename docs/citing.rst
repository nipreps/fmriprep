.. include:: links.rst

.. _citation:

===============
Citing FMRIPREP
===============

Select which options you have run FMRIPREP with to generate custom language
we recommend to include in your paper.


.. raw:: html

   <script language="javascript">
   var version = 'latest';
   function fillCitation(){
      $('#fmriprep_version').text(version);
      $('#workflow_url').text('https://fmriprep.readthedocs.io/en/' + version + '/workflows.html');
      $('#workflow_url').attr('href', 'https://fmriprep.readthedocs.io/en/' + version + '/workflows.html');

      function cb(err, zenodoID) {
         getCitation(zenodoID, 'vancouver-brackets-no-et-al', function(err, citation) {
            $('#fmriprep_citation').text(citation);
         });
         getDOI(zenodoID, function(err, DOI) {
            $('#fmriprep_doi_url').text('https://doi.org/' + DOI);
            $('#fmriprep_doi_url').attr('href', 'https://doi.org/' + DOI);
         });
      }

      if(version == 'latest') {
         getLatestIDFromconceptID("852659", cb);
      } else {
         getZenodoIDFromTag("852659", version, cb);
      }
   }

   function toggle() {
        var controlElementsIds = ["freesurfer","slicetime", "AROMA", "ss_template", "SDC"];
        var controlElementsIdsLength = controlElementsIds.length;

        for (var i = 0; i < controlElementsIdsLength; i++) {
            controlElement = document.getElementById(controlElementsIds[i])

            if (controlElement.checked == null){
                var value = controlElement.value;
            } else {
          	    var value = controlElement.checked;
            }

            $("span[class^='" + controlElementsIds[i] + "_text_']").each(function (i, el) {
                el.style.display='none'
            });

            $("span[class='" + controlElementsIds[i] + "_text_" + value + "']").each(function (i, el) {
                el.style.display='inline'
            });
        }
        return false;
   }
   </script>
   <p>
   With Freesurfer: <input id="freesurfer" type="checkbox" checked="true" onclick="toggle();"/><br />
   Suceptibility Distortion Correction: <select id="SDC" onclick="toggle();">
     <option value="none" selected="true">none</option>
     <option value="TOPUP">TOPUP</option>
     <option value="FUGUE">FUGUE</option>
     <option value="SyN">SyN</option>
   </select><br />
   With AROMA: <input id="AROMA" type="checkbox" onclick="toggle();"/><br />
   Skullstrip template: <select id="ss_template" onclick="toggle();">
     <option value="OASIS" selected="true">OASIS</option>
     <option value="NKI">NKI</option>
   </select><br />
   With slicetime correction: <input id="slicetime" type="checkbox" checked="true" onclick="toggle();"/><br />
   </p>

   <p style="font-style: italic;">
     Results included in this manuscript come from preprocessing performed using
     FMRIPREP version <span id="fmriprep_version">latest</span> [1, 2, RRID:SCR_016216], a Nipype [3, 4, RRID:SCR_002502] based tool.
     Each T1w (T1-weighted) volume was corrected for INU (intensity non-uniformity) using
     <code>N4BiasFieldCorrection</code> v2.1.0 [5] and skull-stripped using
     <code>antsBrainExtraction.sh</code> v2.1.0 (using the
     <span class="ss_template_text_OASIS">OASIS</span>
     <span class="ss_template_text_NKI" style="display: none">NKI</span> template).
     <span class="freesurfer_text_true">Brain surfaces were reconstructed using
     <code>recon-all</code> from FreeSurfer v6.0.1 [6, RRID:SCR_001847],
     and the brain mask estimated previously was refined with a custom variation of
     <a href="https://doi.org/10.1371/journal.pcbi.1005350.g002">the method to reconcile
     ANTs-derived and FreeSurfer-derived segmentations of the cortical gray-matter</a>
     of Mindboggle [21, RRID:SCR_002438].</span>
     Spatial normalization to the ICBM 152 Nonlinear Asymmetrical template version 2009c [7, RRID:SCR_008796]
     was performed through nonlinear registration with the <code>antsRegistration</code>
     tool of ANTs v2.1.0 [8, RRID:SCR_004757], using brain-extracted versions of both T1w volume and template.
     Brain tissue segmentation of cerebrospinal fluid (CSF), white-matter (WM) and
     gray-matter (GM) was performed on the brain-extracted T1w using
     <code>fast</code> [17] (FSL v5.0.9, RRID:SCR_002823).
   </p>

   <p style="font-style: italic;">
     Functional data was <span class="slicetime_text_true">slice time corrected using
     <code>3dTshift</code> from AFNI v16.2.07 [11, RRID:SCR_005927]
     and </span>motion corrected using <code>mcflirt</code> (FSL v5.0.9 [9]).
     <span class="SDC_text_TOPUP" style="display: none">Distortion correction was performed
     using an implementation of the TOPUP technique [10] using <code>3dQwarp</code> (AFNI v16.2.07 [11]).</span>
     <span class="SDC_text_FUGUE" style="display: none">Distortion correction was performed using fieldmaps
     processed with <code>fugue</code> [12] (FSL v5.0.9).</span>
     <span class="SDC_text_SyN" style="display: none">
     "Fieldmap-less" distortion correction was performed by co-registering the functional image to
     the same-subject T1w image with intensity inverted [13,14] constrained with an average fieldmap
     template [15], implemented with <code>antsRegistration</code> (ANTs).</span>
     This was followed by co-registration to the corresponding T1w using boundary-based registration [16]
     with 9 degrees of freedom, using
     <span class="freesurfer_text_true"><code>bbregister</code> (FreeSurfer v6.0.1).</span>
     <span class="freesurfer_text_false" style="display: none"><code>flirt</code> (FSL).</span>
     Motion correcting transformations,
     <span class="SDC_text_TOPUP" style="display: none">field distortion correcting warp, </span>
     <span class="SDC_text_FUGUE" style="display: none">field distortion correcting warp, </span>
     <span class="SDC_text_SyN" style="display: none">field distortion correcting warp, </span>
     BOLD-to-T1w transformation and T1w-to-template (MNI) warp were concatenated and applied in
     a single step using <code>antsApplyTransforms</code> (ANTs v2.1.0) using Lanczos interpolation.
   </p>

   <p style="font-style: italic;">
     Physiological noise regressors were extracted applying CompCor [18].
     Principal components were estimated for the two CompCor variants:
     temporal (tCompCor) and anatomical (aCompCor).
     A mask to exclude signal with cortical origin was obtained by eroding
     the brain mask, ensuring it only contained subcortical structures.
     Six tCompCor components were then calculated including only the top 5% variable
     voxels within that subcortical mask.
     For aCompCor, six components were calculated within the intersection of the
     subcortical mask and the union of CSF and WM masks calculated in T1w space,
     after their projection to the native space of each functional run.
     Frame-wise displacement [19] was calculated for each functional run using the implementation
     of Nipype.
     <span class="AROMA_text_true" style="display: none">ICA-based Automatic Removal Of Motion
     Artifacts (AROMA) was used to generate aggressive noise regressors as well as to create a
     variant of data that is non-aggressively denoised [20].</span>
   </p>

   <p style="font-style: italic;">
     Many internal operations of FMRIPREP use Nilearn [22, RRID:SCR_001362], principally within the BOLD-processing
     workflow.
     For more details of the pipeline see
     <a id="workflow_url" href="http://fmriprep.readthedocs.io/en/latest/workflows.html">http://fmriprep.readthedocs.io/en/latest/workflows.html</a>.
   </p>
   <p>
     1. Esteban O, Markiewicz CJ, Blair RW, Moodie CA, Isik AI, Erramuzpe A, Kent JD, Goncalves M,
        DuPre E, Snyder M, Oya H, Ghosh SS, Wright J, Durnez J, Poldrack RA, Gorgolewski KJ.
        fMRIPrep: a robust preprocessing pipeline for functional MRI.
        Nat Meth. 2018; doi:<a href="https://doi.org/10.1038/s41592-018-0235-4">10.1038/s41592-018-0235-4</a>
   </p>
   <p>
     2. <span id="fmriprep_citation">fMRIPrep</span> Available from: <a id="fmriprep_doi_url" href="https://doi.org/10.5281/zenodo.852659">10.5281/zenodo.852659</a>.
     <img src onerror='fillCitation()' alt="" />
   </p>
   <p>
     3. 	Gorgolewski K, Burns CD, Madison C, Clark D, Halchenko YO, Waskom ML, Ghosh SS. Nipype: a flexible, lightweight and extensible neuroimaging data processing framework in python. Front Neuroinform. 2011 Aug 22;5(August):13. doi:<a href="https://doi.org/10.3389/fninf.2011.00013">10.3389/fninf.2011.00013</a>.
   </p>
   <p>
     4. 	Gorgolewski KJ, Esteban O, Ellis DG, Notter MP, Ziegler E, Johnson H, Hamalainen C, Yvernault B, Burns C, Manhães-Savio A, Jarecka D, Markiewicz CJ, Salo T, Clark D, Waskom M, Wong J, Modat M, Dewey BE, Clark MG, Dayan M, Loney F, Madison C, Gramfort A, Keshavan A, Berleant S, Pinsard B, Goncalves M, Clark D, Cipollini B, Varoquaux G, Wassermann D, Rokem A, Halchenko YO, Forbes J, Moloney B, Malone IB, Hanke M, Mordom D, Buchanan C, Pauli WM, Huntenburg JM, Horea C, Schwartz Y, Tungaraza R, Iqbal S, Kleesiek J, Sikka S, Frohlich C, Kent J, Perez-Guevara M, Watanabe A, Welch D, Cumba C, Ginsburg D, Eshaghi A, Kastman E, Bougacha S, Blair R, Acland B, Gillman A, Schaefer A, Nichols BN, Giavasis S, Erickson D, Correa C, Ghayoor A, Küttner R, Haselgrove C, Zhou D, Craddock RC, Haehn D, Lampe L, Millman J, Lai J, Renfro M, Liu S, Stadler J, Glatard T, Kahn AE, Kong X-Z, Triplett W, Park A, McDermottroe C, Hallquist M, Poldrack R, Perkins LN, Noel M, Gerhard S, Salvatore J, Mertz F, Broderick W, Inati S, Hinds O, Brett M, Durnez J, Tambini A, Rothmei S, Andberg SK, Cooper G, Marina A, Mattfeld A, Urchs S, Sharp P, Matsubara K, Geisler D, Cheung B, Floren A, Nickson T, Pannetier N, Weinstein A, Dubois M, Arias J, Tarbert C, Schlamp K, Jordan K, Liem F, Saase V, Harms R, Khanuja R, Podranski K, Flandin G, Papadopoulos Orfanos D, Schwabacher I, McNamee D, Falkiewicz M, Pellman J, Linkersdörfer J, Varada J, Pérez-García F, Davison A, Shachnev D, Ghosh S. Nipype: a flexible, lightweight and extensible neuroimaging data processing framework in Python. 2017. doi:<a href="https://doi.org/10.5281/zenodo.581704">10.5281/zenodo.581704</a>.
   </p>
   <p>
     5. 	Tustison NJ, Avants BB, Cook PA, Zheng Y, Egan A, Yushkevich PA, Gee JC. N4ITK: improved N3 bias correction. IEEE Trans Med Imaging. 2010 Jun;29(6):1310–20. doi:<a href="https://doi.org/10.1109/TMI.2010.2046908">10.1109/TMI.2010.2046908</a>.
   </p>
   <p>
     6. 	Dale A, Fischl B, Sereno MI. Cortical Surface-Based Analysis: I. Segmentation and Surface Reconstruction. Neuroimage. 1999;9(2):179–94. doi:<a href="https://doi.org/10.1006/nimg.1998.0395">10.1006/nimg.1998.0395</a>.
   </p>
   <p>
     7. 	Fonov VS, Evans AC, McKinstry RC, Almli CR, Collins DL. Unbiased nonlinear average age-appropriate brain templates from birth to adulthood. NeuroImage; Amsterdam. 2009 Jul 1;47:S102. doi:<a href="https://doi.org/10.1016/S1053-8119(09)70884-5">10.1016/S1053-8119(09)70884-5</a>.
   </p>
   <p>
     8. 	Avants BB, Epstein CL, Grossman M, Gee JC. Symmetric diffeomorphic image registration with cross-correlation: evaluating automated labeling of elderly and neurodegenerative brain. Med Image Anal. 2008 Feb;12(1):26–41. doi:<a href="https://doi.org/10.1016/j.media.2007.06.004">10.1016/j.media.2007.06.004</a>.
   </p>
   <p>
     9. 	Jenkinson M, Bannister P, Brady M, Smith S. Improved optimization for the robust and accurate linear registration and motion correction of brain images. Neuroimage. 2002 Oct;17(2):825–41. doi:<a href="https://doi.org/10.1006/nimg.2002.1132">10.1006/nimg.2002.1132</a>.
   </p>
   <p>
     10. 	Andersson JLR, Skare S, Ashburner J. How to correct susceptibility distortions in spin-echo echo-planar images: application to diffusion tensor imaging. Neuroimage. 2003 Oct;20(2):870–88. doi:<a href="https://doi.org/10.1016/S1053-8119(03)00336-7">10.1016/S1053-8119(03)00336-7</a>.
   </p>
   <p>
     11. 	Cox RW. AFNI: software for analysis and visualization of functional magnetic resonance neuroimages. Comput Biomed Res. 1996 Jun;29(3):162–73. doi:<a href="https://doi.org/10.1006/cbmr.1996.0014">10.1006/cbmr.1996.0014</a>.
    </p>
    <p>
     12. 	Jenkinson M. Fast, automated, N-dimensional phase-unwrapping algorithm. Magn Reson Med. 2003 Jan;49(1):193–7. doi:<a href="https://doi.org/10.1002/mrm.10354">10.1002/mrm.10354</a>.
   </p>
   <p>
     13. 	Huntenburg JM. Evaluating nonlinear coregistration of BOLD EPI and T1w images. Freie Universität Berlin; 2014. Available from: <a href="http://hdl.handle.net/11858/00-001M-0000-002B-1CB5-A">http://hdl.handle.net/11858/00-001M-0000-002B-1CB5-A</a>.
   </p>
   <p>
     14. 	Wang S, Peterson DJ, Gatenby JC, Li W, Grabowski TJ, Madhyastha TM. Evaluation of Field Map and Nonlinear Registration Methods for Correction of Susceptibility Artifacts in Diffusion MRI. Front Neuroinform. 2017 [cited 2017 Feb 21];11. doi:<a href="https://doi.org/10.3389/fninf.2017.00017">10.3389/fninf.2017.00017</a>.
   </p>
   <p>
     15. 	Treiber JM, White NS, Steed TC, Bartsch H, Holland D, Farid N, McDonald CR, Carter BS, Dale AM, Chen CC. Characterization and Correction of Geometric Distortions in 814 Diffusion Weighted Images. PLoS One. 2016 Mar 30;11(3):e0152472. doi:<a href="https://doi.org/10.1371/journal.pone.0152472">10.1371/journal.pone.0152472</a>.
   </p>
   <p>
     16. 	Greve DN, Fischl B. Accurate and robust brain image alignment using boundary-based registration. Neuroimage. 2009 Oct;48(1):63–72. doi:<a href="https://doi.org/10.1016/j.neuroimage.2009.06.060">10.1016/j.neuroimage.2009.06.060</a>.
   </p>
   <p>
     17. 	Zhang Y, Brady M, Smith S. Segmentation of brain MR images through a hidden Markov random field model and the expectation-maximization algorithm. IEEE Trans Med Imaging. 2001 Jan;20(1):45–57. doi:<a href="https://doi.org/10.1109/42.906424">10.1109/42.906424</a>.
   </p>
   <p>
     18. 	Behzadi Y, Restom K, Liau J, Liu TT. A component based noise correction method (CompCor) for BOLD and perfusion based fMRI. Neuroimage. 2007 Aug 1;37(1):90–101. doi:<a href="https://doi.org/10.1016/j.neuroimage.2007.04.042">10.1016/j.neuroimage.2007.04.042</a>.
   </p>
   <p>
     19. 	Power JD, Mitra A, Laumann TO, Snyder AZ, Schlaggar BL, Petersen SE. Methods to detect, characterize, and remove motion artifact in resting state fMRI. Neuroimage. 2013 Aug 29;84:320–41. doi:<a href="https://doi.org/10.1016/j.neuroimage.2013.08.048">10.1016/j.neuroimage.2013.08.048</a>.
   </p>
   <p>
     20. 	Pruim RHR, Mennes M, van Rooij D, Llera A, Buitelaar JK, Beckmann CF. ICA-AROMA: A robust ICA-based strategy for removing motion artifacts from fMRI data. Neuroimage. 2015 May 15;112:267–77. doi:<a href="https://doi.org/10.1016/j.neuroimage.2015.02.064">10.1016/j.neuroimage.2015.02.064</a>.
   </p>
   <p>
     21.   Klein A, Ghosh SS, Bao FS, Giard J, Häme Y, Stavsky E, et al. Mindboggling morphometry of human brains.
     PLoS Comput Biol 13(2): e1005350. 2017.
     doi:<a href="https://doi.org/10.1371/journal.pcbi.1005350">10.1371/journal.pcbi.1005350</a>.
   </p>
   <p>
     22.   Abraham A, Pedregosa F, Eickenberg M, Gervais P, Mueller A, Kossaifi J, Gramfort A,
     Thirion B, Varoquaux G. Machine learning for neuroimaging with scikit-learn. Front in Neuroinf 8:14.
     2014. doi:<a href="https://doi.org/10.3389/fninf.2014.00014">10.3389/fninf.2014.00014</a>.
   </p>


Posters
-------

* Organization for Human Brain Mapping 2018
  (`Abstract <https://ww5.aievolution.com/hbm1801/index.cfm?do=abs.viewAbs&abs=1321>`__;
  `PDF <https://files.aievolution.com/hbm1801/abstracts/31779/2035_Markiewicz.pdf>`__)

.. image:: _static/OHBM2018-poster_thumb.png
   :target: _static/OHBM2018-poster.png

* Organization for Human Brain Mapping 2017
  (`Abstract <https://ww5.aievolution.com/hbm1701/index.cfm?do=abs.viewAbs&abs=4111>`__;
  `PDF <https://f1000research.com/posters/6-1129>`__)

.. image:: _static/OHBM2017-poster_thumb.png
   :target: _static/OHBM2017-poster.png

Presentations
-------------

* Organization for Human Brain Mapping 2018
  `Software Demonstration <https://effigies.github.io/fmriprep-demo/>`__.

.. include:: license.rst

Other relevant references
-------------------------

  .. [Fonov2011] Fonov VS, Evans AC, Botteron K, Almli CR, McKinstry RC, Collins DL and BDCG,
      Unbiased average age-appropriate atlases for pediatric studies, NeuroImage 54(1), 2011
      doi:`10.1016/j.neuroimage.2010.07.033 <https://doi.org/10.1016/j.neuroimage.2010.07.033>`_.

  .. [Power2017] Power JD, Plitt M, Kundu P, Bandettini PA, Martin A (2017) Temporal interpolation alters
      motion in fMRI scans: Magnitudes and consequences for artifact detection. PLOS ONE 12(9): e0182939.
      doi:`10.1371/journal.pone.0182939 <https://doi.org/10.1371/journal.pone.0182939>`_.

  .. [Brett2001] Brett M, Leff AP, Rorden C, Ashburner J (2001) Spatial Normalization of Brain Images with
      Focal Lesions Using Cost Function Masking. NeuroImage 14(2)
      doi:`10.006/nimg.2001.0845 <https://doi.org/10.1006/nimg.2001.0845>`_.
