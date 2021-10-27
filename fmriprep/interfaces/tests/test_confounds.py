from nipype.pipeline import engine as pe
from fmriprep.interfaces import confounds
from pathlib import Path


def test_RenameACompCor(tmp_path, data_dir):
    renamer = pe.Node(confounds.RenameACompCor(), name="renamer",
                      base_dir=str(tmp_path))
    renamer.inputs.components_file = data_dir / "acompcor_truncated.tsv"
    renamer.inputs.metadata_file = data_dir / "component_metadata_truncated.tsv"

    res = renamer.run()

    target_components = Path.read_text(data_dir / "acompcor_renamed.tsv")
    target_meta = Path.read_text(data_dir / "component_metadata_renamed.tsv")
    renamed_components = Path(res.outputs.components_file).read_text()
    renamed_meta = Path(res.outputs.metadata_file).read_text()
    assert renamed_components == target_components
    assert renamed_meta == target_meta


def test_FilterDropped(tmp_path, data_dir):
    filt = pe.Node(confounds.FilterDropped(), name="filt", base_dir=str(tmp_path))
    filt.inputs.in_file = data_dir / "component_metadata_truncated.tsv"

    res = filt.run()

    target_meta = Path.read_text(data_dir / "component_metadata_filtered.tsv")
    filtered_meta = Path(res.outputs.out_file).read_text()
    assert filtered_meta == target_meta
