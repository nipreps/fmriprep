from packaging.version import LegacyVersion
from pathlib import Path
import shutil

from nipype.interfaces import fsl
import pytest
import templateflow.api as tf


fslversion = fsl.Info.version()
TEMPLATE = tf.get("MNI152NLin2009cAsym", resolution=2, desc=None, suffix="T1w")


@pytest.mark.skipif(fslversion is None, reason="fsl required")
@pytest.mark.skipif(LegacyVersion(fslversion) < LegacyVersion("6.0.0"), reason="FSL6 test")
@pytest.mark.parametrize("path_parent,filename", [
    (".", "brain.nii.gz"),
    (
        "pneumonoultramicroscopicsilicovolcanoconiosis/floccinaucinihilipilification",
        "supercalifragilisticexpialidocious.nii.gz",
    ),
    (
        "pneumonoultramicroscopicsilicovolcanoconiosis/floccinaucinihilipilification/"
        "antidisestablishmentarianism/pseudopseudohypoparathyroidism/sesquipedalian",
        "brain.nii.gz"
    )
])
def test_fsl6_long_filenames(tmp_path, path_parent, filename):
    test_dir = tmp_path / path_parent
    test_dir.mkdir(parents=True, exist_ok=True)
    in_file = test_dir / filename
    out_file = test_dir / "output.nii.gz"
    shutil.copy(TEMPLATE, in_file)

    bet = fsl.BET(in_file=in_file, out_file=out_file).run()
    assert Path(bet.outputs.out_file).exists()
