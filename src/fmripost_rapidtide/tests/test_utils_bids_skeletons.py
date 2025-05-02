"""Test data collection using skeleton-based datasets."""

import pytest
from bids.layout import BIDSLayout
from niworkflows.utils.testing import generate_bids_skeleton

from fmripost_rapidtide.tests.utils import check_expected, get_test_data_path
from fmripost_rapidtide.utils import bids as xbids


def test_collect_derivatives_longitudinal_01(tmpdir):
    """Test collect_derivatives with a mocked up longitudinal dataset."""
    # Generate a BIDS dataset
    bids_dir = tmpdir / 'collect_derivatives_longitudinal_01'
    dset_yaml = str(get_test_data_path() / 'skeletons' / 'do_longitudinal_01.yml')
    generate_bids_skeleton(str(bids_dir), dset_yaml)

    layout = BIDSLayout(bids_dir, config=['bids', 'derivatives'], validate=False)

    subject_data = xbids.collect_derivatives(
        raw_dataset=None,
        derivatives_dataset=layout,
        entities={'subject': '102'},
        fieldmap_id=None,
        spec=None,
        patterns=None,
        allow_multiple=True,
    )
    expected = {
        'bold_native': [
            'sub-102_ses-1_task-rest_desc-preproc_bold.nii.gz',
            'sub-102_ses-2_task-rest_desc-preproc_bold.nii.gz',
        ],
    }
    check_expected(subject_data, expected)


def test_collect_derivatives_longitudinal_02(tmpdir):
    """Test collect_derivatives with a mocked up longitudinal dataset."""
    # Generate a BIDS dataset
    bids_dir = tmpdir / 'collect_derivatives_longitudinal_02'
    dset_yaml = str(get_test_data_path() / 'skeletons' / 'do_longitudinal_02.yml')
    generate_bids_skeleton(str(bids_dir), dset_yaml)

    layout = BIDSLayout(bids_dir, config=['bids', 'derivatives'], validate=False)

    # Query for all sessions should return all bold derivatives
    subject_data = xbids.collect_derivatives(
        raw_dataset=None,
        derivatives_dataset=layout,
        entities={'subject': '102'},
        fieldmap_id=None,
        spec=None,
        patterns=None,
        allow_multiple=True,
    )
    expected = {
        'bold_native': [
            'sub-102_ses-1_task-rest_desc-preproc_bold.nii.gz',
            'sub-102_ses-2_task-rest_desc-preproc_bold.nii.gz',
            'sub-102_ses-3_task-rest_desc-preproc_bold.nii.gz',
        ],
    }
    check_expected(subject_data, expected)

    # Query for session 2 should return anat from session 2
    subject_data = xbids.collect_derivatives(
        raw_dataset=None,
        derivatives_dataset=layout,
        entities={'subject': '102', 'session': '2'},
        fieldmap_id=None,
        spec=None,
        patterns=None,
        allow_multiple=False,
    )
    expected = {
        'anat_dseg': 'sub-102_ses-2_dseg.nii.gz',
    }
    check_expected(subject_data, expected)

    # Query for session 3 (no anat available)
    with pytest.raises(
        ValueError,
        match='Multiple anatomical derivatives found',
    ):
        xbids.collect_derivatives(
            raw_dataset=None,
            derivatives_dataset=layout,
            entities={'subject': '102', 'session': '3'},
            fieldmap_id=None,
            spec=None,
            patterns=None,
            allow_multiple=False,
        )


def test_collect_derivatives_longitudinal_03(tmpdir):
    """Test collect_derivatives with a mocked up longitudinal dataset."""
    # Generate a BIDS dataset
    bids_dir = tmpdir / 'collect_derivatives_longitudinal_03'
    dset_yaml = str(get_test_data_path() / 'skeletons' / 'do_longitudinal_03.yml')
    generate_bids_skeleton(str(bids_dir), dset_yaml)

    layout = BIDSLayout(bids_dir, config=['bids', 'derivatives'], validate=False)

    # Query for session 1 should return anat from session 1
    subject_data = xbids.collect_derivatives(
        raw_dataset=None,
        derivatives_dataset=layout,
        entities={'subject': '102', 'session': '1'},
        fieldmap_id=None,
        spec=None,
        patterns=None,
        allow_multiple=False,
    )
    expected = {
        'anat_dseg': 'sub-102_ses-1_dseg.nii.gz',
    }
    check_expected(subject_data, expected)

    # Query for session 2 should return anat from session 1 if no anat is present for session 2
    # XXX: Currently this doesn't work.
    subject_data = xbids.collect_derivatives(
        raw_dataset=None,
        derivatives_dataset=layout,
        entities={'subject': '102', 'session': '2'},
        fieldmap_id=None,
        spec=None,
        patterns=None,
        allow_multiple=False,
    )
    expected = {
        'anat_dseg': 'sub-102_ses-1_dseg.nii.gz',
    }
    check_expected(subject_data, expected)


def test_collect_derivatives_xsectional_04(tmpdir):
    """Test collect_derivatives with a mocked up cross-sectional dataset."""
    # Generate a BIDS dataset
    bids_dir = tmpdir / 'collect_derivatives_xsectional_04'
    dset_yaml = str(get_test_data_path() / 'skeletons' / 'do_crosssectional_01.yml')
    generate_bids_skeleton(str(bids_dir), dset_yaml)

    layout = BIDSLayout(bids_dir, config=['bids', 'derivatives'], validate=False)

    # Query for subject 102 should return anat from subject 102,
    # even though there are multiple subjects with anat derivatives,
    # because the subject is specified in the entities dictionary.
    subject_data = xbids.collect_derivatives(
        raw_dataset=None,
        derivatives_dataset=layout,
        entities={'subject': '102'},
        fieldmap_id=None,
        spec=None,
        patterns=None,
        allow_multiple=False,
    )
    expected = {
        'anat_dseg': 'sub-102_dseg.nii.gz',
    }
    check_expected(subject_data, expected)


def test_collect_rawderiv_xsectional(tmpdir):
    """Test collect_derivatives with a mocked up cross-sectional dataset."""
    # Generate a BIDS dataset
    bids_dir = tmpdir / 'collect_rawderiv_xsectional'
    deriv_dir = bids_dir / 'derivatives' / 'fmriprep'
    raw_yaml = str(get_test_data_path() / 'skeletons' / 'r_crosssectional_01.yml')
    deriv_yaml = str(get_test_data_path() / 'skeletons' / 'd_crosssectional_01.yml')
    generate_bids_skeleton(str(bids_dir), raw_yaml)
    generate_bids_skeleton(str(deriv_dir), deriv_yaml)
    raw_layout = BIDSLayout(bids_dir, config=['bids'], derivatives=False, validate=False)
    deriv_layout = BIDSLayout(deriv_dir, config=['bids', 'derivatives'], validate=False)

    # Query for subject 102 should return anat from subject 102,
    # even though there are multiple subjects with anat derivatives,
    # because the subject is specified in the entities dictionary.
    subject_data = xbids.collect_derivatives(
        raw_dataset=raw_layout,
        derivatives_dataset=deriv_layout,
        entities={'subject': '102'},
        fieldmap_id=None,
        spec=None,
        patterns=None,
        allow_multiple=True,
    )
    expected = {
        'bold_raw': ['sub-102_task-rest_bold.nii.gz'],
        'bold_native': ['sub-102_task-rest_desc-preproc_bold.nii.gz'],
    }
    check_expected(subject_data, expected)

    # Query for subject 102 should return anat from subject 102,
    # even though there are multiple subjects with anat derivatives,
    # because the subject is specified in the entities dictionary.
    run_data = xbids.collect_derivatives(
        raw_dataset=raw_layout,
        derivatives_dataset=deriv_layout,
        entities={'subject': '102'},
        fieldmap_id='funcpepolar01',
        spec=None,
        patterns=None,
        allow_multiple=False,
    )
    expected = {
        'bold_raw': 'sub-102_task-rest_bold.nii.gz',
        'bold_native': 'sub-102_task-rest_desc-preproc_bold.nii.gz',
        'bold_mask_native': 'sub-102_task-rest_desc-brain_mask.nii.gz',
        'boldref': 'sub-102_task-rest_desc-coreg_boldref.nii.gz',
        'confounds': 'sub-102_task-rest_desc-confounds_timeseries.tsv',
        'anat_dseg': 'sub-102_dseg.nii.gz',
        'hmc': 'sub-102_task-rest_from-orig_to-boldref_mode-image_desc-hmc_xfm.txt',
        'boldref2anat': 'sub-102_task-rest_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt',
        'boldref2fmap': 'sub-102_task-rest_from-boldref_to-funcpepolar01_mode-image_xfm.txt',
    }
    check_expected(run_data, expected)


def test_collect_rawderiv_xsectional_spaces(tmpdir):
    """Test collect_derivatives with a mocked up cross-sectional dataset."""
    from niworkflows.utils.spaces import Reference, SpatialReferences

    # Generate a BIDS dataset
    bids_dir = tmpdir / 'collect_rawderiv_xsectional'
    deriv_dir = bids_dir / 'derivatives' / 'fmriprep'
    raw_yaml = str(get_test_data_path() / 'skeletons' / 'r_crosssectional_01.yml')
    deriv_yaml = str(get_test_data_path() / 'skeletons' / 'd_crosssectional_01.yml')
    generate_bids_skeleton(str(bids_dir), raw_yaml)
    generate_bids_skeleton(str(deriv_dir), deriv_yaml)
    raw_layout = BIDSLayout(bids_dir, config=['bids'], derivatives=False, validate=False)
    deriv_layout = BIDSLayout(deriv_dir, config=['bids', 'derivatives'], validate=False)

    # Query for subject 102 should return anat from subject 102,
    # even though there are multiple subjects with anat derivatives,
    # because the subject is specified in the entities dictionary.
    subject_data = xbids.collect_derivatives(
        raw_dataset=raw_layout,
        derivatives_dataset=deriv_layout,
        entities={'subject': '102'},
        fieldmap_id=None,
        spec=None,
        patterns=None,
        allow_multiple=True,
    )
    expected = {
        'bold_raw': ['sub-102_task-rest_bold.nii.gz'],
        'bold_native': ['sub-102_task-rest_desc-preproc_bold.nii.gz'],
    }
    check_expected(subject_data, expected)

    # Query for subject 102 should return anat from subject 102,
    # even though there are multiple subjects with anat derivatives,
    # because the subject is specified in the entities dictionary.
    run_data = xbids.collect_derivatives(
        raw_dataset=raw_layout,
        derivatives_dataset=deriv_layout,
        entities={'subject': '102'},
        fieldmap_id='funcpepolar01',
        spec=None,
        patterns=None,
        allow_multiple=False,
        spaces=SpatialReferences([Reference('MNI152NLin6Asym')]),
    )
    expected = {
        'bold_raw': 'sub-102_task-rest_bold.nii.gz',
        'bold_native': 'sub-102_task-rest_desc-preproc_bold.nii.gz',
        'bold_mask_native': 'sub-102_task-rest_desc-brain_mask.nii.gz',
        'boldref': 'sub-102_task-rest_desc-coreg_boldref.nii.gz',
        'confounds': 'sub-102_task-rest_desc-confounds_timeseries.tsv',
        'anat_dseg': 'sub-102_dseg.nii.gz',
        'hmc': 'sub-102_task-rest_from-orig_to-boldref_mode-image_desc-hmc_xfm.txt',
        'boldref2anat': 'sub-102_task-rest_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt',
        'boldref2fmap': 'sub-102_task-rest_from-boldref_to-funcpepolar01_mode-image_xfm.txt',
        'anat2outputspaces': ['sub-102_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5'],
    }
    check_expected(run_data, expected)
