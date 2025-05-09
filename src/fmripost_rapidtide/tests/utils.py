"""Utility functions for tests."""

import lzma
import os
import subprocess
import tarfile
from contextlib import contextmanager
from glob import glob
from io import BytesIO
from pathlib import Path

import requests
from nipype import logging

LOGGER = logging.getLogger('nipype.utils')


def get_nodes(wf_results):
    """Load nodes from a Nipype workflow's results."""
    return {node.fullname: node for node in wf_results.nodes}


def download_test_data(data_dir=None):
    """Download test data from 28&He.

    If data_dir is provided, the outputs will be in data_dir/fmriprep-testing.
    Otherwise they will be downloaded into a copy of the package.

    The test data has been pre-processed with FMRIPREP and reduced to 100 volumes.
    The spatial resolution has been reduced to 3mm voxels.

    This also downloads the raw bids BOLD for the same session. It has
    been resampled to 3mm isotropic voxels as well.

    """
    URLS = {
        '28+he_metadata': 'https://upenn.box.com/shared/static/widyfgj9yb9e8j5kbiq9h78fuwybtc4d.xz',
        '28+he_bold': 'https://upenn.box.com/shared/static/ieeehyqbt40d5fjdfzr63xnchbaoum9g.xz',
        '28+he_anat': 'https://upenn.box.com/shared/static/t0i1kb6ecglniljrs4up55pfb5gdv655.xz',
        '28+he_bids': 'https://upenn.box.com/shared/static/na8xayoluuqb8bn81kss9a92l0ro6t9c.xz',
    }

    if not data_dir:
        data_dir = os.path.join(os.path.dirname(get_test_data_path()), 'test_data')
    out_dir = os.path.join(data_dir, 'fmriprep-testing')

    if os.path.isdir(out_dir):
        LOGGER.info(
            'Dataset already exists. '
            'If you need to re-download the data, please delete the folder.'
        )
        return out_dir
    else:
        LOGGER.info(f'Downloading to {data_dir}')

    os.makedirs(out_dir, exist_ok=True)
    for dset_name, url in URLS.items():
        LOGGER.info(f'Downloading {dset_name} from {url}')
        with requests.get(url, stream=True) as req:
            with lzma.open(BytesIO(req.content)) as xz:
                with tarfile.open(fileobj=xz) as t:
                    t.extractall(data_dir)

    return out_dir


def check_generated_files(output_dir, output_list_file):
    """Compare files generated by xcp_d with a list of expected files."""
    found_files = sorted(glob(os.path.join(output_dir, '**/*'), recursive=True))
    found_files = [os.path.relpath(f, output_dir) for f in found_files]

    # Ignore figures
    found_files = [f for f in found_files if 'figures' not in f]

    # Ignore logs
    found_files = [f for f in found_files if 'log' not in f.split(os.path.sep)]

    with open(output_list_file) as fileobj:
        expected_files = fileobj.readlines()
        expected_files = [f.rstrip() for f in expected_files]

    if sorted(found_files) != sorted(expected_files):
        expected_not_found = sorted(set(expected_files) - set(found_files))
        found_not_expected = sorted(set(found_files) - set(expected_files))

        msg = ''
        if expected_not_found:
            msg += '\nExpected but not found:\n\t'
            msg += '\n\t'.join(expected_not_found)

        if found_not_expected:
            msg += '\nFound but not expected:\n\t'
            msg += '\n\t'.join(found_not_expected)
        raise ValueError(msg)


def run_command(command, env=None):
    """Run a given shell command with certain environment variables set."""
    merged_env = os.environ
    if env:
        merged_env.update(env)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
        env=merged_env,
    )
    while True:
        line = process.stdout.readline()
        line = str(line, 'utf-8')[:-1]
        print(line)
        if line == '' and process.poll() is not None:
            break

    if process.returncode != 0:
        raise Exception(
            f'Non zero return code: {process.returncode}\n{command}\n\n{process.stdout.read()}'
        )


@contextmanager
def chdir(path):
    """Temporarily change directories.

    Taken from https://stackoverflow.com/a/37996581/2589328.
    """
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)


def reorder_expected_outputs():
    """Load each of the expected output files and sort the lines alphabetically.

    This function is called manually by devs when they modify the test outputs.
    """
    test_data_path = get_test_data_path()
    expected_output_files = sorted(glob(os.path.join(test_data_path, 'test_*_outputs.txt')))
    for expected_output_file in expected_output_files:
        LOGGER.info(f'Sorting {expected_output_file}')

        with open(expected_output_file) as fileobj:
            file_contents = fileobj.readlines()

        file_contents = sorted(set(file_contents))

        with open(expected_output_file, 'w') as fileobj:
            fileobj.writelines(file_contents)


def list_files(startpath):
    """List files in a directory."""
    tree = ''
    for root, _, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        tree += f'{indent}{os.path.basename(root)}/\n'
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            tree += f'{subindent}{f}\n'

    return tree


@contextmanager
def modified_environ(*remove, **update):
    """
    Temporarily updates the ``os.environ`` dictionary in-place.

    The ``os.environ`` dictionary is updated in-place so that the modification
    is sure to work in all situations.

    :param remove: Environment variables to remove.
    :param update: Dictionary of environment variables and values to add/update.
    """
    env = os.environ
    update = update or {}
    remove = remove or []

    # List of environment variables being updated or removed.
    stomped = (set(update.keys()) | set(remove)) & set(env.keys())
    # Environment variables and values to restore on exit.
    update_after = {k: env[k] for k in stomped}
    # Environment variables and values to remove on exit.
    remove_after = frozenset(k for k in update if k not in env)

    try:
        env.update(update)
        [env.pop(k, None) for k in remove]
        yield
    finally:
        env.update(update_after)
        [env.pop(k) for k in remove_after]


def get_test_data_path():
    """Return the path to test datasets, terminated with separator.

    Test-related data are kept in tests folder in 'data'.
    Based on function by Yaroslav Halchenko used in Neurosynth Python package.
    """
    return Path(__file__).resolve().parent.parent.parent.parent / 'tests' / 'data'


def check_expected(subject_data, expected):
    """Check expected values."""
    for key, value in expected.items():
        if isinstance(value, str):
            assert subject_data[key] is not None, f'Key {key} is None.'
            if os.path.basename(subject_data[key]) != value:
                raise AssertionError(f'{os.path.basename(subject_data[key])} != {value}')
        elif isinstance(value, list):
            assert subject_data[key] is not None, f'Key {key} is None.'
            if len(subject_data[key]) != len(value):
                raise AssertionError(f'Key {key} expected {value}, got {subject_data[key]}')

            for item, expected_item in zip(subject_data[key], value, strict=False):
                if os.path.basename(item) != expected_item:
                    raise AssertionError(f'{os.path.basename(item)} != {expected_item}')
        else:
            assert subject_data[key] is value
