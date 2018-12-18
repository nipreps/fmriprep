# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Temporary patches
-----------------

"""

from random import randint
from time import sleep

from numpy.linalg.linalg import LinAlgError
from nipype.algorithms import confounds as nac
from nipype.interfaces.base import File
from nipype.interfaces.mixins import reporting


class RetryCompCorInputSpecMixin(reporting.ReportCapableInputSpec):
    out_report = File('report.html', usedefault=True, hash_files=False,
                      desc='filename for warning HTML snippet')


class RetryCompCorMixin(reporting.ReportCapableInterface):
    def _run_interface(self, runtime):
        warn = self.inputs.failure_mode == 'NaN'

        failures = 0
        save_exc = None
        while True:
            success = True
            # Identifiy success/failure in both error and NaN mode
            try:
                runtime = super()._run_interface(runtime)
                if warn and self._is_allnans():
                    success = False
            except LinAlgError as exc:
                success = False
                save_exc = exc

            if success:
                break

            failures += 1
            if failures > 10:
                if warn:
                    break
                raise save_exc
            start = (failures - 1) * 10
            sleep(randint(start + 4, start + 10))

        return runtime

    def _is_allnans(self):
        import numpy as np
        outputs = self._list_outputs()
        components = np.loadtxt(outputs['components_file'], skiprows=1)
        return np.isnan(components).all()

    def _generate_report(self):
        snippet = '<!-- {} completed without error -->'.format(self._header)
        if self._is_allnans():
            snippet = '''\
<p class="elem-desc">
    Warning: {} components could not be estimated, due to a linear algebra error.
    While not definitive, this may be an indication of a poor mask.
    Please inspect the {} contours above to ensure that they are located
    in the white matter/CSF.
</p>
'''.format(self._header, 'magenta' if self._header[0] == 'a' else 'blue')

        with open(self._out_report, 'w') as fobj:
            fobj.write(snippet)


class RobustACompCorInputSpec(RetryCompCorInputSpecMixin, nac.CompCorInputSpec):
    pass


class RobustACompCorOutputSpec(reporting.ReportCapableOutputSpec, nac.CompCorOutputSpec):
    pass


class RobustACompCor(RetryCompCorMixin, nac.ACompCor):
    """
    Runs aCompCor several times if it suddenly fails with
    https://github.com/poldracklab/fmriprep/issues/776

    Warns by default, rather than failing, on linear algebra errors.
    https://github.com/poldracklab/fmriprep/issues/1433

    """
    input_spec = RobustACompCorInputSpec
    output_spec = RobustACompCorOutputSpec


class RobustTCompCorInputSpec(RetryCompCorInputSpecMixin, nac.TCompCorInputSpec):
    pass


class RobustTCompCorOutputSpec(reporting.ReportCapableOutputSpec, nac.TCompCorOutputSpec):
    pass


class RobustTCompCor(RetryCompCorMixin, nac.TCompCor):
    """
    Runs tCompCor several times if it suddenly fails with
    https://github.com/poldracklab/fmriprep/issues/776

    Warns by default, rather than failing, on linear algebra errors.
    https://github.com/poldracklab/fmriprep/issues/1433

    """
    input_spec = RobustTCompCorInputSpec
    output_spec = RobustTCompCorOutputSpec
