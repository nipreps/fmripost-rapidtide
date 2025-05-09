"""Lightweight tests for fmripost_rapidtide.utils.bids."""

import pytest
from bids.layout import BIDSLayout, BIDSLayoutIndexer

from fmripost_rapidtide.tests.utils import check_expected, get_test_data_path
from fmripost_rapidtide.utils import bids as xbids


def test_collect_derivatives_raw(base_ignore_list):
    """Test collect_derivatives with a raw dataset."""
    data_dir = get_test_data_path()

    raw_dataset = data_dir / 'ds000005-fmriprep' / 'sourcedata' / 'raw'
    raw_layout = BIDSLayout(
        raw_dataset,
        config=['bids'],
        validate=False,
        indexer=BIDSLayoutIndexer(validate=False, index_metadata=False, ignore=base_ignore_list),
    )

    subject_data = xbids.collect_derivatives(
        raw_dataset=raw_layout,
        derivatives_dataset=None,
        entities={'subject': '01', 'task': 'mixedgamblestask'},
        fieldmap_id=None,
        spec=None,
        patterns=None,
        allow_multiple=True,
    )
    expected = {
        'bold_raw': [
            'sub-01_task-mixedgamblestask_run-01_bold.nii.gz',
            'sub-01_task-mixedgamblestask_run-02_bold.nii.gz',
        ],
    }
    check_expected(subject_data, expected)

    with pytest.raises(ValueError, match='Multiple files found'):
        xbids.collect_derivatives(
            raw_dataset=raw_layout,
            derivatives_dataset=None,
            entities={'subject': '01', 'task': 'mixedgamblestask'},
            fieldmap_id=None,
            spec=None,
            patterns=None,
            allow_multiple=False,
        )


def test_collect_derivatives_minimal(minimal_ignore_list):
    """Test collect_derivatives with a minimal-mode dataset."""
    data_dir = get_test_data_path()

    derivatives_dataset = data_dir / 'ds000005-fmriprep'
    derivatives_layout = BIDSLayout(
        derivatives_dataset,
        config=['bids', 'derivatives'],
        validate=False,
        indexer=BIDSLayoutIndexer(
            validate=False,
            index_metadata=False,
            ignore=minimal_ignore_list,
        ),
    )

    subject_data = xbids.collect_derivatives(
        raw_dataset=None,
        derivatives_dataset=derivatives_layout,
        entities={'subject': '01', 'task': 'mixedgamblestask', 'run': 1},
        fieldmap_id=None,
        spec=None,
        patterns=None,
    )
    expected = {
        'bold_native': None,
        'bold_mask_native': None,
        # TODO: Add bold_mask_native to the dataset
        # 'bold_mask_native': 'sub-01_task-mixedgamblestask_run-01_desc-brain_mask.nii.gz',
        'bold_confounds': [],
        'bold_hmc': [],
        'boldref2anat': (
            'sub-01_task-mixedgamblestask_run-01_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt'
        ),
        'boldref2fmap': None,
    }
    check_expected(subject_data, expected)


def test_collect_derivatives_full(full_ignore_list):
    """Test collect_derivatives with a full-mode dataset."""
    data_dir = get_test_data_path()

    derivatives_dataset = data_dir / 'ds000005-fmriprep'
    derivatives_layout = BIDSLayout(
        derivatives_dataset,
        config=['bids', 'derivatives'],
        validate=False,
        indexer=BIDSLayoutIndexer(validate=False, index_metadata=False, ignore=full_ignore_list),
    )

    subject_data = xbids.collect_derivatives(
        raw_dataset=None,
        derivatives_dataset=derivatives_layout,
        entities={'subject': '01', 'task': 'mixedgamblestask', 'run': 1},
        fieldmap_id=None,
        spec=None,
        patterns=None,
    )
    expected = {
        'bold_native': None,
        'bold_mask_native': None,
        'bold_confounds': [],
        'bold_hmc': [],
        'boldref2anat': (
            'sub-01_task-mixedgamblestask_run-01_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt'
        ),
        'boldref2fmap': None,
    }
    check_expected(subject_data, expected)
