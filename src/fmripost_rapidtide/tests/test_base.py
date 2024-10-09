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
        config.workflow.ampthresh = -1.0
        config.workflow.autorespdelete = False
        config.workflow.autosync = False
        config.workflow.bipolar = False
        config.workflow.confoundderiv = True
        config.workflow.confoundpowers = 1
        config.workflow.corrweighting = 'regressor'
        config.workflow.detrendorder = 1
        config.workflow.filterband = 'lfo'
        config.workflow.glmderivs = 1
        config.workflow.globalpcacomponents = 0.8
        config.workflow.globalsignalmethod = 'sum'
        config.workflow.lagmaxthresh = 5.0
        config.workflow.lagminthresh = 0.5
        config.workflow.maxpasses = 1
        config.workflow.numnull = 100
        config.workflow.numtozero = 0
        config.workflow.outputlevel = 'min'
        config.workflow.pcacomponents = 0.8
        config.workflow.sigmalimit = 1000.0
        config.workflow.searchrange = [-30, 30]
        config.workflow.sigmathresh = 100.0
        config.workflow.simcalcrange = [-1, -1]
        config.workflow.spatialfilt = 4.0
        config.workflow.timerange = [-1, -1]

        wf = init_rapidtide_fit_wf(
            bold_file='sub-01_task-rest_bold.nii.gz',
            metadata={'RepetitionTime': 2.0},
            mem_gb={
                'resampled': 1,
                'largemem': 2,
            },
        )
        assert wf.name == 'rapidtide_task_rest_wf'
