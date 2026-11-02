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
"""Tests for descriptions.tsv generation."""

from __future__ import annotations

import json

import pytest

from fmriprep.data import load as load_data
from fmriprep.utils.bids import write_descriptions_tsv


def test_descriptions_json_loads():
    """Test that descriptions.json loads correctly."""
    desc_data = json.loads(load_data.readable('descriptions.json').read_text())

    assert '_version' in desc_data
    assert 'entities' in desc_data
    assert len(desc_data['entities']) > 0

    # Check that required entities exist
    required_entities = ['preproc', 'brain', 'hmc', 'coreg', 'confounds']
    for entity in required_entities:
        assert entity in desc_data['entities'], f'Missing required entity: {entity}'


def test_descriptions_json_structure():
    """Test that each entity in descriptions.json has correct structure."""
    desc_data = json.loads(load_data.readable('descriptions.json').read_text())

    for entity_id, entity_def in desc_data['entities'].items():
        # Each entity must have a 'base' description
        assert 'base' in entity_def, f"Entity '{entity_id}' missing 'base' field"
        assert isinstance(entity_def['base'], str), f"Entity '{entity_id}' base must be string"

        # Parameters must be a list if present
        if 'parameters' in entity_def:
            assert isinstance(entity_def['parameters'], list), (
                f"Entity '{entity_id}' parameters must be list"
            )

        # Conditional parts must have correct structure if present
        if 'conditional_parts' in entity_def:
            assert isinstance(entity_def['conditional_parts'], list), (
                f"Entity '{entity_id}' conditional_parts must be list"
            )
            for part in entity_def['conditional_parts']:
                assert 'condition' in part, f"Entity '{entity_id}' part missing condition"
                assert 'text' in part, f"Entity '{entity_id}' part missing text"


def test_write_descriptions_tsv_basic(tmp_path):
    """Test basic descriptions.tsv generation."""
    tsv_path = write_descriptions_tsv(tmp_path)

    assert tsv_path.exists()
    assert tsv_path.name == 'descriptions.tsv'

    # Read and parse TSV
    content = tsv_path.read_text()
    lines = content.strip().split('\n')

    # Check header
    assert lines[0] == 'desc_id\tdescription\tparameters'

    # Check that we have data rows
    assert len(lines) > 1

    # Parse rows
    for line in lines[1:]:
        parts = line.split('\t')
        assert len(parts) == 3, f'Each row should have 3 columns: {line}'
        _desc_id, _description, parameters = parts

        # Check parameters is valid JSON
        json.loads(parameters)


def test_write_descriptions_tsv_with_sdc(tmp_path):
    """Test descriptions.tsv with SDC enabled."""
    tsv_path = write_descriptions_tsv(
        tmp_path,
        sdc_method='TOPUP',
        slice_timing_corrected=True,
        slice_time_ref=0.5,
    )

    content = tsv_path.read_text()

    # Find preproc row
    for line in content.split('\n'):
        if line.startswith('preproc\t'):
            parts = line.split('\t')
            description = parts[1]
            parameters = json.loads(parts[2])

            # SDC should be mentioned in description
            assert 'susceptibility distortion correction' in description.lower()
            assert 'TOPUP' in description

            # Slice timing should be mentioned
            assert 'slice timing correction' in description.lower()

            # Check parameters
            assert parameters['sdc_applied'] is True
            assert parameters['stc_applied'] is True
            assert parameters['sdc_method'] == 'TOPUP'
            break
    else:
        pytest.fail('preproc row not found')


def test_write_descriptions_tsv_without_sdc(tmp_path):
    """Test descriptions.tsv without SDC (SDC not applied)."""
    tsv_path = write_descriptions_tsv(
        tmp_path,
        sdc_method=None,  # No SDC
        slice_timing_corrected=False,
    )

    content = tsv_path.read_text()

    # Find preproc row
    for line in content.split('\n'):
        if line.startswith('preproc\t'):
            parts = line.split('\t')
            description = parts[1]
            parameters = json.loads(parts[2])

            # SDC should NOT be mentioned in description
            assert 'susceptibility distortion correction' not in description.lower()

            # Slice timing should NOT be mentioned
            assert 'slice timing correction' not in description.lower()

            # Check parameters
            assert parameters['sdc_applied'] is False
            assert parameters['stc_applied'] is False
            break
    else:
        pytest.fail('preproc row not found')


def test_write_descriptions_tsv_freesurfer_mask(tmp_path):
    """Test that FreeSurfer mask source is reflected in description."""
    tsv_path = write_descriptions_tsv(tmp_path, freesurfer=True)

    content = tsv_path.read_text()

    for line in content.split('\n'):
        if line.startswith('brain\t'):
            description = line.split('\t')[1]
            assert 'FreeSurfer' in description
            break


def test_write_descriptions_tsv_ants_mask(tmp_path):
    """Test that ANTs mask source is reflected in description."""
    tsv_path = write_descriptions_tsv(tmp_path, freesurfer=False)

    content = tsv_path.read_text()

    for line in content.split('\n'):
        if line.startswith('brain\t'):
            description = line.split('\t')[1]
            assert 'ANTs' in description
            break


def test_write_descriptions_tsv_confounds(tmp_path):
    """Test confounds description includes thresholds when provided."""
    tsv_path = write_descriptions_tsv(
        tmp_path,
        fd_threshold=0.5,
        dvars_threshold=1.5,
        compcor_variance=0.5,
    )

    content = tsv_path.read_text()

    for line in content.split('\n'):
        if line.startswith('confounds\t'):
            parts = line.split('\t')
            description = parts[1]
            parameters = json.loads(parts[2])

            # Check description mentions thresholds
            assert 'FD' in description
            assert '0.5' in description
            assert 'DVARS' in description
            assert '1.5' in description

            # Check parameters
            assert parameters['fd_threshold'] == 0.5
            assert parameters['dvars_threshold'] == 1.5
            break


def test_write_descriptions_tsv_coreg_bbr(tmp_path):
    """Test coregistration description with BBR."""
    tsv_path = write_descriptions_tsv(
        tmp_path,
        coreg_method='bbr',
        bold2anat_dof=6,
    )

    content = tsv_path.read_text()

    for line in content.split('\n'):
        if line.startswith('coreg\t'):
            description = line.split('\t')[1]
            assert 'boundary-based registration' in description
            assert '6' in description
            assert 'degrees of freedom' in description
            break


def test_write_descriptions_tsv_all_entities_present(tmp_path):
    """Test that all defined entities are present in output."""
    desc_data = json.loads(load_data.readable('descriptions.json').read_text())
    expected_entities = set(desc_data['entities'].keys())

    tsv_path = write_descriptions_tsv(tmp_path)
    content = tsv_path.read_text()

    found_entities = set()
    for line in content.split('\n')[1:]:  # Skip header
        if line.strip():
            desc_id = line.split('\t')[0]
            found_entities.add(desc_id)

    assert found_entities == expected_entities, (
        f'Missing entities: {expected_entities - found_entities}, '
        f'Extra entities: {found_entities - expected_entities}'
    )
