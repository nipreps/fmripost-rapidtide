# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2023 The NiPreps Developers <nipreps@gmail.com>
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
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""fMRIPost-rapidtide workflows to run Rapidtide."""

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe

from fmripost_rapidtide import config
from fmripost_rapidtide.interfaces.rapidtide import Rapidtide
from fmripost_rapidtide.utils.utils import _get_wf_name


def init_rapidtide_wf(
    *,
    bold_file: str,
    metadata: dict,
    mem_gb: dict,
):
    """Build a workflow that runs `Rapidtide`_.

    This workflow wraps `Rapidtide`_ to characterize and remove the traveling wave artifact.

    The following steps are performed:

    #. Remove non-steady state volumes from the bold series.
    #. Run rapidtide
    #. Collect rapidtide outputs
    #. Generate a confounds file with the rapidtide outputs

    .. _Rapidtide: https://rapidtide.readthedocs.io/

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fmripost_rapidtide.workflows.bold.confounds import init_rapidtide_wf

            wf = init_rapidtide_wf(
                bold_file="fake.nii.gz",
                metadata={"RepetitionTime": 1.0},
            )

    Parameters
    ----------
    bold_file
        BOLD series used as name source for derivatives
    metadata : :obj:`dict`
        BIDS metadata for BOLD file

    Inputs
    ------
    bold
        BOLD series in template space
    bold_mask
        BOLD series mask in template space
    dseg
        Tissue segmentation in template space
    confounds
        fMRIPrep-formatted confounds file, which must include the following columns:
        "trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z".
    skip_vols
        number of non steady state volumes

    Outputs
    -------
    denoised_bold
    confounds_file
    """

    from nipype.interfaces.base import Undefined
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    from fmripost_rapidtide.interfaces.bids import DerivativesDataSink
    from fmripost_rapidtide.interfaces.nilearn import SplitDseg

    workflow = Workflow(name=_get_wf_name(bold_file, 'rapidtide'))
    workflow.__postdesc__ = """\
Identification and removal of traveling wave artifacts was performed using rapidtide.
"""

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'bold',
                'bold_mask',
                'dseg',
                'confounds',
                'skip_vols',
            ],
        ),
        name='inputnode',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'delay_map',
                'regressor_file',
            ],
        ),
        name='outputnode',
    )

    # Split tissue-type segmentation to get GM and WM masks
    split_tissues = pe.Node(
        SplitDseg(),
        name='split_tissues',
    )
    workflow.connect([(inputnode, split_tissues, [('dseg', 'dseg')])])

    # Run the Rapidtide classifier
    # XXX: simcalcrange is converted to list of strings
    rapidtide = pe.Node(
        Rapidtide(
            outputname='rapidtide',
            datatstep=metadata['RepetitionTime'],
            autosync=config.workflow.autosync,
            filterband=config.workflow.filterband,
            filterfreqs=config.workflow.filterfreqs or Undefined,
            filterstopfreqs=config.workflow.filterstopfreqs or Undefined,
            numnull=config.workflow.numnull,
            detrendorder=config.workflow.detrendorder,
            spatialfilt=config.workflow.spatialfilt,
            confoundfile=config.workflow.confoundfile or Undefined,
            confoundpowers=config.workflow.confoundpowers,
            confoundderiv=config.workflow.confoundderiv,
            globalsignalmethod=config.workflow.globalsignalmethod,
            globalpcacomponents=config.workflow.globalpcacomponents,
            numtozero=config.workflow.numtozero,
            timerange=[int(float(i)) for i in config.workflow.timerange],
            corrweighting=config.workflow.corrweighting,
            simcalcrange=[int(float(i)) for i in config.workflow.simcalcrange],
            fixdelay=config.workflow.fixdelay or Undefined,
            searchrange=[int(float(i)) for i in config.workflow.searchrange],
            sigmalimit=config.workflow.sigmalimit,
            bipolar=config.workflow.bipolar,
            lagminthresh=config.workflow.lagminthresh,
            lagmaxthresh=config.workflow.lagmaxthresh,
            ampthresh=config.workflow.ampthresh,
            sigmathresh=config.workflow.sigmathresh,
            pcacomponents=config.workflow.pcacomponents,
            convergencethresh=config.workflow.convergencethresh or Undefined,
            maxpasses=config.workflow.maxpasses,
            glmsourcefile=config.workflow.glmsourcefile or Undefined,
            glmderivs=config.workflow.glmderivs,
            outputlevel=config.workflow.outputlevel,
            territorymap=config.workflow.territorymap or Undefined,
            autorespdelete=config.workflow.autorespdelete,
            nprocs=config.nipype.omp_nthreads,
        ),
        name='rapidtide',
        mem_gb=mem_gb['filesize'] * 6,
        n_procs=config.nipype.omp_nthreads,
    )
    workflow.connect([
        (inputnode, rapidtide, [
            ('bold', 'in_file'),
            ('bold_mask', 'brainmask'),
            ('confounds', 'motionfile'),
            ('skip_vols', 'numskip'),
        ]),
        (split_tissues, rapidtide, [
            ('gm', 'graymattermask'),
            ('wm', 'whitemattermask'),
            ('gm', 'globalmeaninclude'),  # GM mask for initial regressor selection
            ('gm', 'refineinclude'),  # GM mask for refinement
            ('gm', 'offsetinclude'),  # GM mask for offset calculation
        ]),
    ])  # fmt:skip

    ds_delay_map = pe.Node(
        DerivativesDataSink(
            compress=True,
            desc='delay',
            suffix='boldmap',
        ),
        name='ds_delay_map',
        run_without_submitting=True,
    )
    workflow.connect([
        (rapidtide, ds_delay_map, [
            ('maxtimemap', 'in_file'),
            ('maxtimemap_json', 'meta_dict'),
        ]),
        (ds_delay_map, outputnode, [('out_file', 'delay_map')]),
    ])  # fmt:skip

    ds_regressor = pe.Node(
        DerivativesDataSink(
            desc='regressor',
            suffix='timeseries',
            extension='.tsv',
        ),
        name='ds_regressor',
        run_without_submitting=True,
    )
    workflow.connect([
        (rapidtide, ds_regressor, [
            ('lagtcgenerator', 'in_file'),
            ('lagtcgenerator_json', 'meta_dict'),
        ]),
        (ds_regressor, outputnode, [('out_file', 'regressor_file')]),
    ])  # fmt:skip

    return workflow
