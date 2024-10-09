"""Tests for fmripost_rapidtide.workflows."""

from fmriprep.workflows.tests import mock_config

from fmripost_rapidtide import config


def test_init_rapidtide_fit_wf(tmp_path_factory):
    from fmripost_rapidtide.workflows.rapidtide import init_rapidtide_fit_wf

    tempdir = tmp_path_factory.mktemp('test_init_rapidtide_fit_wf')

    with mock_config():
        config.execution.output_dir = tempdir / 'out'
        config.execution.work_dir = tempdir / 'work'
        config.workflow.err_on_warn = False
        config.workflow.timerange = [-1, -1]
        config.workflow.simcalcrange = [-1, -1]
        config.workflow.searchrange = [-30, 30]

        wf = init_rapidtide_fit_wf(
            bold_file='sub-01_task-rest_bold.nii.gz',
            metadata={'RepetitionTime': 2.0},
            mem_gb={
                'resampled': 1,
                'largemem': 2,
            },
        )
        assert wf.name == 'rapidtide_task_rest_wf'
