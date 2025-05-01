"""Test data collection using skeleton-based datasets."""

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
    subject_data = xbids.collect_derivatives(
        raw_dataset=None,
        derivatives_dataset=layout,
        entities={'subject': '102', 'session': '3'},
        fieldmap_id=None,
        spec=None,
        patterns=None,
        allow_multiple=False,
    )
    expected = {
        'anat_dseg': None,
    }
    check_expected(subject_data, expected)


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
        'anat_dseg': None,
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
