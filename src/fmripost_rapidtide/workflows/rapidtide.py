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
from fmripost_rapidtide.utils.utils import _get_wf_name, load_json


def init_rapidtide_fit_wf(
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

            from fmripost_rapidtide.workflows.rapidtide import init_rapidtide_fit_wf

            wf = init_rapidtide_fit_wf(
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
    from fmripost_rapidtide.interfaces.rapidtide import Rapidtide
    from fmripost_rapidtide.workflows.outputs import init_rapidtide_map_reporting_wf

    workflow = Workflow(name=_get_wf_name(bold_file, 'rapidtide'))
    workflow.__postdesc__ = """\
Identification and removal of traveling wave artifacts was performed using rapidtide.
"""

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'bold',
                'boldref',
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
                'rapidtide_root',
                'delay_map',
                'lagtcgenerator',
                'valid_mask',
                'run_options',
                'strength_map',
                'slfo_amplitude',
                'runoptions',
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

    # Run Rapidtide
    # XXX: simcalcrange is converted to list of strings
    rapidtide = pe.Node(
        Rapidtide(
            ampthresh=config.workflow.ampthresh,
            autorespdelete=config.workflow.autorespdelete,
            autosync=config.workflow.autosync,
            bipolar=config.workflow.bipolar,
            noconfoundderiv=config.workflow.noconfoundderiv,
            confoundfile=config.workflow.confoundfile or Undefined,
            confoundpowers=config.workflow.confoundpowers,
            convergencethresh=config.workflow.convergencethresh or Undefined,
            corrweighting=config.workflow.corrweighting,
            datatstep=metadata['RepetitionTime'],
            detrendorder=config.workflow.detrendorder,
            filterband=config.workflow.filterband,
            filterfreqs=config.workflow.filterfreqs or Undefined,
            filterstopfreqs=config.workflow.filterstopfreqs or Undefined,
            fixdelay=config.workflow.fixdelay or Undefined,
            regressderivs=config.workflow.regressderivs,
            glmsourcefile=config.workflow.glmsourcefile or Undefined,
            globalpcacomponents=config.workflow.globalpcacomponents,
            globalsignalmethod=config.workflow.globalsignalmethod,
            lagmaxthresh=config.workflow.lagmaxthresh,
            lagminthresh=config.workflow.lagminthresh,
            maxpasses=config.workflow.maxpasses,
            nprocs=config.nipype.omp_nthreads,
            numnull=config.workflow.numnull,
            numskip=config.workflow.numskip,
            outputlevel=config.workflow.outputlevel,
            pcacomponents=config.workflow.pcacomponents,
            searchrange=[int(float(i)) for i in config.workflow.searchrange],
            sigmalimit=config.workflow.sigmalimit,
            sigmathresh=config.workflow.sigmathresh,
            simcalcrange=[int(float(i)) for i in config.workflow.simcalcrange],
            spatialfilt=config.workflow.spatialfilt,
            territorymap=config.workflow.territorymap or Undefined,
            timerange=[int(float(i)) for i in config.workflow.timerange],
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
            ('csf', 'csfmask'),
            ('gm', 'globalmeaninclude'),  # GM mask for initial regressor selection
        ]),
        (rapidtide, outputnode, [
            ('maskfile', 'valid_mask'),
            ('runoptions', 'runoptions'),
        ]),
    ])  # fmt:skip

    combine_prefix = pe.Node(
        niu.Function(_combine_prefix),
        name='combine_prefix',
    )
    workflow.connect([
        (rapidtide, combine_prefix, [
            ('rapidtide_dir', 'in1'),
            ('prefix', 'in2'),
        ]),
        (combine_prefix, [('out', 'rapidtide_root')]),
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
            (('maxtimemap_json', load_json), 'meta_dict'),
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
            (('lagtcgenerator_json', load_json), 'meta_dict'),
        ]),
        (ds_regressor, outputnode, [('out_file', 'lagtcgenerator')]),
    ])  # fmt:skip

    ds_strength_map = pe.Node(
        DerivativesDataSink(
            compress=True,
            desc='strength',
            suffix='boldmap',
        ),
        name='ds_strength_map',
        run_without_submitting=True,
    )
    workflow.connect([
        (rapidtide, ds_strength_map, [
            ('strengthmap', 'in_file'),
            (('strengthmap_json', load_json), 'meta_dict'),
        ]),
        (ds_strength_map, outputnode, [('out_file', 'strength_map')]),
    ])  # fmt:skip

    ds_slfo_amplitude = pe.Node(
        DerivativesDataSink(
            compress=True,
            desc='sLFOamplitude',
            suffix='timeseries',
        ),
        name='ds_slfo_amplitude',
        run_without_submitting=True,
    )
    workflow.connect([
        (rapidtide, ds_slfo_amplitude, [
            ('slfoamplitude', 'in_file'),
            (('slfoamplitude_json', load_json), 'meta_dict'),
        ]),
        (ds_slfo_amplitude, outputnode, [('out_file', 'slfo_amplitude')]),
    ])  # fmt:skip

    map_dict = {
        'maxtimemap': ['Delay Map', 'delay', 'viridis'],
        'strengthmap': ['Strength Map', 'strength', 'hot'],
        'delayrankordermap': ['Delay Rank Order Map', 'timepercentile', 'viridis'],
        'correlationwidthmap': ['Correlation Width Map', 'maxwidth', 'plasma'],
    }
    for image, (title, metric, cmap) in map_dict.items():
        report_wf = init_rapidtide_map_reporting_wf(
            title=title,
            metric=metric,
            cmap=cmap,
            name=f'{image}_report_wf',
        )
        workflow.connect([
            (inputnode, report_wf, [('boldref', 'inputnode.boldref')]),
            (rapidtide, report_wf, [
                (image, 'inputnode.in_file'),
                ('maskfile', 'inputnode.mask'),
            ]),
        ])  # fmt:skip

    return workflow


def init_rapidtide_denoise_wf(
    *,
    bold_file: str,
    metadata: dict,
    mem_gb: dict,
):
    """Build a workflow that runs `Rapidtide`_.

    XXX: UNUSED

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

            from fmripost_rapidtide.workflows.rapidtide import init_rapidtide_fit_wf

            wf = init_rapidtide_fit_wf(
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

    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    from fmripost_rapidtide.interfaces.bids import DerivativesDataSink
    from fmripost_rapidtide.interfaces.rapidtide import RetroRegress

    workflow = Workflow(name=_get_wf_name(bold_file, 'denoise'))
    workflow.__postdesc__ = """\
Identification and removal of traveling wave artifacts was performed using rapidtide.
"""

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'bold',
                'bold_mask',
                'rapidtide_root',
                'skip_vols',
            ],
        ),
        name='inputnode',
    )

    # Remove the traveling wave artifact
    retro_regress = pe.Node(
        RetroRegress(
            nprocs=config.nipype.omp_nthreads,
            regressderivs=config.workflow.regressderivs,
        ),
        name='retro_regress',
        mem_gb=mem_gb['filesize'] * 6,
        n_procs=config.nipype.omp_nthreads,
    )
    workflow.connect([
        (inputnode, retro_regress, [
            ('bold', 'in_file'),
            ('bold_mask', 'brainmask'),
            ('rapidtide_root', 'datafileroot'),
            ('skip_vols', 'numskip'),
        ]),
    ])  # fmt:skip

    ds_denoised_bold = pe.Node(
        DerivativesDataSink(
            compress=True,
            desc='denoised',
            suffix='bold',
        ),
        name='ds_denoised_bold',
        run_without_submitting=True,
    )
    workflow.connect([
        (retro_regress, ds_denoised_bold, [
            ('denoised', 'in_file'),
            (('denoised_json', load_json), 'meta_dict'),
        ]),
    ])  # fmt:skip

    return workflow


def init_rapidtide_confounds_wf(
    *,
    bold_file: str,
    metadata: dict,
    mem_gb: dict,
):
    """Generate rapidtide confounds from rapidtide derivatives.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fmripost_rapidtide.workflows.rapidtide import init_rapidtide_fit_wf

            wf = init_rapidtide_fit_wf(
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
        BOLD series in native space
    bold_mask
        BOLD series mask in native space
    dseg
        Tissue segmentation in native space
    confounds
        fMRIPrep-formatted confounds file, which must include the following columns:
        "trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z".
    skip_vols
        number of non steady state volumes

    Outputs
    -------
    voxelwise_regressor
        The 4D lagged sLFO regressor file.
    """

    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    from fmripost_rapidtide.interfaces.bids import DerivativesDataSink
    from fmripost_rapidtide.interfaces.rapidtide import RetroLagTCS

    workflow = Workflow(name=_get_wf_name(bold_file, 'confounds'))
    workflow.__postdesc__ = """\
Identification and removal of traveling wave artifacts was performed using rapidtide.
"""

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'bold',
                'bold_mask',
                'valid_mask',  # desc-corrfit_mask
                'delay_map',
                'lagtcgenerator',
                'skip_vols',
                'anat_dseg',
                'boldref2anat',
                'anat2outputspaces',
            ],
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'voxelwise_regressor',
            ]
        ),
        name='outputnode',
    )

    # Generate the traveling wave artifact voxel-wise regressor
    retrolagtcs = pe.Node(
        RetroLagTCS(
            nprocs=config.nipype.omp_nthreads,
        ),
        name='retrolagtcs',
        mem_gb=mem_gb['filesize'] * 6,
        n_procs=config.nipype.omp_nthreads,
    )
    workflow.connect([
        (inputnode, retrolagtcs, [
            ('bold', 'in_file'),
            ('valid_mask', 'maskfile'),
            ('delay_map', 'lagtimesfile'),
            ('lagtcgenerator', 'lagtcgeneratorfile'),
            ('skip_vols', 'numskip'),
        ]),
    ])  # fmt:skip

    ds_voxelwise_regressor = pe.Node(
        DerivativesDataSink(
            compress=True,
            desc='sLFO',
            suffix='timeseries',
        ),
        name='ds_voxelwise_regressor',
        run_without_submitting=True,
    )
    workflow.connect([
        (retrolagtcs, ds_voxelwise_regressor, [
            ('filter_file', 'in_file'),
            (('filter_json', load_json), 'meta_dict'),
        ]),
        (ds_voxelwise_regressor, outputnode, [('out_file', 'voxelwise_regressor')])
    ])  # fmt:skip

    return workflow


def _combine_prefix(in1, in2):
    import os

    return os.path.join(in1, in2)
