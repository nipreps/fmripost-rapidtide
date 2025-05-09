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
"""
fMRIPost Rapidtide workflows
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_fmripost_rapidtide_wf
.. autofunction:: init_single_subject_wf

"""

import os
import sys
from collections import defaultdict
from copy import deepcopy

import yaml
from nipype.pipeline import engine as pe
from packaging.version import Version

from fmripost_rapidtide import config
from fmripost_rapidtide.utils.utils import _get_wf_name, update_dict


def init_fmripost_rapidtide_wf():
    """Build *fMRIPost-rapidtide*'s pipeline.

    This workflow organizes the execution of fMRIPost-rapidtide,
    with a sub-workflow for each subject.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fmripost_rapidtide.workflows.tests import mock_config
            from fmripost_rapidtide.workflows.base import init_fmripost_rapidtide_wf

            with mock_config():
                wf = init_fmripost_rapidtide_wf()

    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    ver = Version(config.environment.version)

    fmripost_rapidtide_wf = Workflow(name=f'fmripost_rapidtide_{ver.major}_{ver.minor}_wf')
    fmripost_rapidtide_wf.base_dir = config.execution.work_dir

    for subject_id in config.execution.participant_label:
        single_subject_wf = init_single_subject_wf(subject_id)

        single_subject_wf.config['execution']['crashdump_dir'] = str(
            config.execution.output_dir / f'sub-{subject_id}' / 'log' / config.execution.run_uuid
        )
        for node in single_subject_wf._get_all_nodes():
            node.config = deepcopy(single_subject_wf.config)

        fmripost_rapidtide_wf.add_nodes([single_subject_wf])

        # Dump a copy of the config file into the log directory
        log_dir = (
            config.execution.output_dir / f'sub-{subject_id}' / 'log' / config.execution.run_uuid
        )
        log_dir.mkdir(exist_ok=True, parents=True)
        config.to_filename(log_dir / 'fmripost_rapidtide.toml')

    return fmripost_rapidtide_wf


def init_single_subject_wf(subject_id: str):
    """Organize the postprocessing pipeline for a single subject.

    It collects and reports information about the subject,
    and prepares sub-workflows to postprocess each BOLD series.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fmripost_rapidtide.workflows.tests import mock_config
            from fmripost_rapidtide.workflows.base import init_single_subject_wf

            with mock_config():
                wf = init_single_subject_wf('01')

    Parameters
    ----------
    subject_id : :obj:`str`
        Subject label for this single-subject workflow.

    Notes
    -----
    1.  Load fMRIPost-rapidtide config file.
    2.  Collect fMRIPrep derivatives.
        -   BOLD file in native space.
        -   Two main possibilities:
            1.  bids_dir is a raw BIDS dataset and preprocessing derivatives
                are provided through ``--derivatives``.
                In this scenario, we only need minimal derivatives.
            2.  bids_dir is a derivatives dataset and we need to collect compliant
                derivatives to get the data into the right space.
    3.  Loop over runs.
    4.  Collect each run's associated files.
        -   Transform(s) to target spaces
        -   Confounds file
        -   Rapidtide uses its own standard-space edge, CSF, and brain masks,
            so we don't need to worry about those.
    5.  Transform dseg from anat space to boldref space.
    6.  Run Rapidtide on boldref space data.
    7.  Warp derivatives (denoised BOLD, delay map, 4D regressor) to requested output spaces.
        Warp the denoised BOLD from boldref to target instead of warping preprocessed BOLD and then
        denoising.
    8.  Create reportlets.
    """
    from bids.utils import listify
    from nipype.interfaces import utility as niu
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.bids import BIDSInfo
    from niworkflows.interfaces.nilearn import NILEARN_VERSION

    from fmripost_rapidtide.interfaces.bids import DerivativesDataSink
    from fmripost_rapidtide.interfaces.nilearn import MeanImage
    from fmripost_rapidtide.interfaces.reportlets import AboutSummary, SubjectSummary
    from fmripost_rapidtide.utils.bids import collect_derivatives

    spaces = config.workflow.spaces

    workflow = Workflow(name=f'sub_{subject_id}_wf')
    workflow.__desc__ = f"""
Results included in this manuscript come from postprocessing
performed using *fMRIPost-rapidtide* {config.environment.version} (@rapidtide),
which is based on *Nipype* {config.environment.nipype_version}
(@nipype1; @nipype2; RRID:SCR_002502).

"""
    workflow.__postdesc__ = f"""

Many internal operations of *fMRIPost-rapidtide* use
*Nilearn* {NILEARN_VERSION} [@nilearn, RRID:SCR_001362].
For more details of the pipeline, see [the section corresponding
to workflows in *fMRIPost-rapidtide*'s documentation]\
(https://fmripost_rapidtide.readthedocs.io/en/latest/workflows.html \
"FMRIPrep's documentation").


### Copyright Waiver

The above boilerplate text was automatically generated by fMRIPost-rapidtide
with the express intention that users should copy and paste this
text into their manuscripts *unchanged*.
It is released under the [CC0]\
(https://creativecommons.org/publicdomain/zero/1.0/) license.

### References

"""

    if config.execution.derivatives:
        # Raw dataset + derivatives dataset
        config.loggers.workflow.info('Raw+derivatives workflow mode enabled')
        subject_data = collect_derivatives(
            raw_dataset=config.execution.layout,
            derivatives_dataset=None,
            entities=config.execution.bids_filters,
            fieldmap_id=None,
            allow_multiple=True,
            spaces=None,
        )
        subject_data['bold'] = listify(subject_data['bold_raw'])
    else:
        # Derivatives dataset only
        config.loggers.workflow.info('Derivatives-only workflow mode enabled')
        subject_data = collect_derivatives(
            raw_dataset=None,
            derivatives_dataset=config.execution.layout,
            entities=config.execution.bids_filters,
            fieldmap_id=None,
            allow_multiple=True,
            spaces=None,
        )
        # Patch standard-space BOLD files into 'bold' key
        subject_data['bold'] = listify(subject_data['bold_native'])

    # Make sure we always go through these two checks
    if not subject_data['bold']:
        task_id = config.execution.task_id
        raise RuntimeError(
            f'No BOLD images found for participant {subject_id} and '
            f'task {task_id if task_id else "<all>"}. '
            'All workflows require BOLD images. '
            f'Please check your BIDS filters: {config.execution.bids_filters}.'
        )

    bids_info = pe.Node(
        BIDSInfo(
            bids_dir=config.execution.bids_dir,
            bids_validate=False,
            in_file=subject_data['bold'][0],
        ),
        name='bids_info',
    )

    summary = pe.Node(
        SubjectSummary(
            bold=subject_data['bold'],
            std_spaces=spaces.get_spaces(nonstandard=False),
            nstd_spaces=spaces.get_spaces(standard=False),
        ),
        name='summary',
        run_without_submitting=True,
    )
    workflow.connect([(bids_info, summary, [('subject', 'subject_id')])])

    about = pe.Node(
        AboutSummary(version=config.environment.version, command=' '.join(sys.argv)),
        name='about',
        run_without_submitting=True,
    )

    ds_report_summary = pe.Node(
        DerivativesDataSink(
            source_file=subject_data['bold'][0],
            base_directory=config.execution.output_dir,
            desc='summary',
            datatype='figures',
        ),
        name='ds_report_summary',
        run_without_submitting=True,
    )
    workflow.connect([(summary, ds_report_summary, [('out_report', 'in_file')])])

    ds_report_about = pe.Node(
        DerivativesDataSink(
            source_file=subject_data['bold'][0],
            base_directory=config.execution.output_dir,
            desc='about',
            datatype='figures',
        ),
        name='ds_report_about',
        run_without_submitting=True,
    )
    workflow.connect([(about, ds_report_about, [('out_report', 'in_file')])])

    # Append the functional section to the existing anatomical excerpt
    # That way we do not need to stream down the number of bold datasets
    func_pre_desc = f"""
Functional data postprocessing

: For each of the {len(subject_data['bold'])} BOLD runs found per subject
(across all tasks and sessions), the following postprocessing was performed.
"""
    workflow.__desc__ += func_pre_desc

    denoise_within_run = (len(subject_data['bold']) == 1) or not config.workflow.average_over_runs
    if not denoise_within_run:
        # Average the lag map across runs before denoising
        # XXX: This won't actually work, since they aren't in the same boldref space.
        merge_lag_maps = pe.Node(
            niu.Merge(len(subject_data['bold'])),
            name='merge_lag_maps',
        )
        average_lag_map = pe.Node(
            MeanImage(),
            name='average_lag_map',
        )

    for i_run, bold_file in enumerate(subject_data['bold']):
        fit_single_run_wf = init_fit_single_run_wf(bold_file=bold_file)
        denoise_single_run_wf = init_denoise_single_run_wf(bold_file=bold_file)

        workflow.connect([
            (fit_single_run_wf, denoise_single_run_wf, [
                ('outputnode.rapidtide_root', 'inputnode.rapidtide_root'),
                ('outputnode.lagtcgenerator', 'inputnode.lagtcgenerator'),
                # XXX: Need to add valid mask and runoptions to the inputnode
                ('outputnode.valid_mask', 'inputnode.valid_mask'),
                ('outputnode.runoptions', 'inputnode.runoptions'),
                # transforms and related files
                ('outputnode.bold_native', 'inputnode.bold'),
                ('outputnode.bold_mask_native', 'inputnode.bold_mask'),
                ('outputnode.anat_dseg', 'inputnode.anat_dseg'),
                ('outputnode.boldref2anat', 'inputnode.boldref2anat'),
                ('outputnode.anat2outputspaces', 'inputnode.anat2outputspaces'),
                ('outputnode.anat2outputspaces_templates', 'inputnode.templates'),
            ]),
        ])  # fmt:skip

        if denoise_within_run:
            # Denoise the BOLD data using the run-wise lag map
            workflow.connect([
                (fit_single_run_wf, denoise_single_run_wf, [
                    ('outputnode.delay_map', 'inputnode.delay_map'),
                ]),
            ])  # fmt:skip
        else:
            # Denoise the BOLD data using the mean lag map
            workflow.connect([
                (fit_single_run_wf, merge_lag_maps, [('outputnode.delay_map', f'in{i_run + 1}')]),
                (merge_lag_maps, average_lag_map, [('out', 'in_file')]),
                (average_lag_map, denoise_single_run_wf, [('out_file', 'inputnode.delay_map')]),
            ])  # fmt:skip

    return workflow


def init_fit_single_run_wf(*, bold_file):
    """Set up a single-run workflow for fMRIPost-rapidtide."""
    from fmriprep.utils.misc import estimate_bold_mem_usage
    from nipype.interfaces import utility as niu
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    from fmripost_rapidtide.interfaces.misc import ApplyTransforms
    from fmripost_rapidtide.utils.bids import collect_derivatives, extract_entities
    from fmripost_rapidtide.workflows.rapidtide import init_rapidtide_fit_wf

    spaces = config.workflow.spaces
    omp_nthreads = config.nipype.omp_nthreads

    workflow = Workflow(name=_get_wf_name(bold_file, 'single_run'))
    workflow.__desc__ = ''

    bold_metadata = config.execution.layout.get_metadata(bold_file)
    mem_gb = estimate_bold_mem_usage(bold_file)[1]

    entities = extract_entities(bold_file)

    functional_cache = defaultdict(list, {})
    if config.execution.derivatives:
        # Collect native-space derivatives and transforms
        functional_cache = collect_derivatives(
            raw_dataset=config.execution.layout,
            derivatives_dataset=None,
            entities=entities,
            fieldmap_id=None,
            allow_multiple=False,
            spaces=None,
        )
        for deriv_dir in config.execution.derivatives.values():
            functional_cache = update_dict(
                functional_cache,
                collect_derivatives(
                    raw_dataset=None,
                    derivatives_dataset=deriv_dir,
                    entities=entities,
                    fieldmap_id=None,
                    allow_multiple=False,
                    spaces=spaces,
                ),
            )

        if not functional_cache['confounds']:
            if config.workflow.dummy_scans is None:
                raise ValueError(
                    'No confounds detected. '
                    'Automatical dummy scan detection cannot be performed. '
                    'Please set the `--dummy-scans` flag explicitly.'
                )

            # TODO: Calculate motion parameters from motion correction transform
            raise ValueError('Motion parameters cannot be extracted from transforms yet.')

    else:
        # Collect boldref:res-native derivatives
        # Only derivatives dataset was passed in, so we expected boldref-space derivatives
        functional_cache.update(
            collect_derivatives(
                raw_dataset=None,
                derivatives_dataset=config.execution.layout,
                entities=entities,
                fieldmap_id=None,
                allow_multiple=False,
                spaces=spaces,
            ),
        )
        if not functional_cache['bold_native']:
            raise FileNotFoundError('No boldref:res-native BOLD images found.')

    # Now determine whether to use boldref-space derivatives or raw data + transforms

    config.loggers.workflow.info(
        (
            f'Collected run data for {os.path.basename(bold_file)}:\n'
            f'{yaml.dump(functional_cache, default_flow_style=False, indent=4)}'
        ),
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'bold_native',
                'bold_mask_native',
                'anat_dseg',
                'boldref2anat',
                'anat2outputspaces',
                'anat2outputspaces_templates',
                'rapidtide_root',
                'delay_map',
                'lagtcgenerator',
                'valid_mask',
                'runoptions',
            ],
        ),
        name='outputnode',
    )
    outputnode.inputs.anat_dseg = functional_cache['anat_dseg']
    outputnode.inputs.boldref2anat = functional_cache['boldref2anat']
    outputnode.inputs.anat2outputspaces = functional_cache['anat2outputspaces']
    outputnode.inputs.anat2outputspaces_templates = functional_cache['anat2outputspaces_templates']

    if config.workflow.dummy_scans is not None:
        skip_vols = config.workflow.dummy_scans
    else:
        if not functional_cache['confounds']:
            raise ValueError(
                'No confounds detected. '
                'Automatic dummy scan detection cannot be performed. '
                'Please set the `--dummy-scans` flag explicitly.'
            )
        skip_vols = get_nss(functional_cache['confounds'])

    boldref_buffer = pe.Node(
        niu.IdentityInterface(fields=['bold', 'bold_mask']),
        name='boldref_buffer',
    )
    boldref_buffer.inputs.bold_mask = functional_cache['bold_mask_native']

    # Warp the dseg from anatomical space to boldref space
    dseg_to_boldref = pe.Node(
        ApplyTransforms(
            dimension=3,
            interpolation='GenericLabel',
            input_image=functional_cache['anat_dseg'],
            reference_image=functional_cache['boldref'],
            transforms=[functional_cache['boldref2anat']],
            invert_transform_flags=[True],
            num_threads=config.nipype.omp_nthreads,
            args='--verbose',
        ),
        name='dseg_to_boldref',
        mem_gb=mem_gb['filesize'],
        n_procs=config.nipype.omp_nthreads,
    )

    if ('bold_native' not in functional_cache) and ('bold_raw' in functional_cache):
        from fmriprep.workflows.bold.apply import init_bold_volumetric_resample_wf
        from fmriprep.workflows.bold.stc import init_bold_stc_wf
        from niworkflows.interfaces.header import ValidateImage

        workflow.__desc__ += """\
Raw BOLD series were resampled to boldref:res-native, for rapidtide denoising.
"""

        validate_bold = pe.Node(
            ValidateImage(in_file=functional_cache['bold_raw']),
            name='validate_bold',
        )

        stc_buffer = pe.Node(
            niu.IdentityInterface(fields=['bold_file']),
            name='stc_buffer',
        )
        run_stc = ('SliceTiming' in bold_metadata) and 'slicetiming' not in config.workflow.ignore
        if run_stc:
            bold_stc_wf = init_bold_stc_wf(
                mem_gb=mem_gb,
                metadata=bold_metadata,
                name='bold_stc_wf',
            )
            bold_stc_wf.inputs.inputnode.skip_vols = skip_vols
            workflow.connect([
                (validate_bold, bold_stc_wf, [('out_file', 'inputnode.bold_file')]),
                (bold_stc_wf, stc_buffer, [('outputnode.stc_file', 'bold_file')]),
            ])  # fmt:skip
        else:
            workflow.connect([(validate_bold, stc_buffer, [('out_file', 'bold_file')])])

        bold_boldref_wf = init_bold_volumetric_resample_wf(
            metadata=bold_metadata,
            fieldmap_id=None,  # XXX: Ignoring the field map for now
            omp_nthreads=omp_nthreads,
            mem_gb=mem_gb,
            jacobian='fmap-jacobian' not in config.workflow.ignore,
            name='bold_boldref_wf',
        )
        bold_boldref_wf.inputs.inputnode.motion_xfm = functional_cache['hmc']
        bold_boldref_wf.inputs.inputnode.boldref2fmap_xfm = functional_cache['boldref2fmap']
        bold_boldref_wf.inputs.inputnode.resolution = 'native'
        bold_boldref_wf.inputs.inputnode.bold_ref_file = functional_cache['boldref']
        bold_boldref_wf.inputs.inputnode.target_mask = functional_cache['bold_mask_native']
        bold_boldref_wf.inputs.inputnode.target_ref_file = functional_cache['boldref']

        workflow.connect([
            # XXX: Ignoring the field map for now
            # (inputnode, bold_boldref_wf, [
            #     ('fmap_ref', 'inputnode.fmap_ref'),
            #     ('fmap_coeff', 'inputnode.fmap_coeff'),
            #     ('fmap_id', 'inputnode.fmap_id'),
            # ]),
            (stc_buffer, bold_boldref_wf, [('bold_file', 'inputnode.bold_file')]),
            (bold_boldref_wf, boldref_buffer, [('outputnode.bold_file', 'bold')]),
        ])  # fmt:skip

    elif 'bold_native' in functional_cache:
        workflow.__desc__ += """\
Preprocessed BOLD series in boldref:res-native space were collected for rapidtide denoising.
"""
        boldref_buffer.inputs.bold = functional_cache['bold_native']

    else:
        raise ValueError('No valid BOLD series found for rapidtide denoising.')

    workflow.connect([
        (boldref_buffer, outputnode, [
            ('bold', 'bold_native'),
            ('bold_mask', 'bold_mask_native'),
        ]),
    ])  # fmt:skip

    # Run rapidtide
    rapidtide_wf = init_rapidtide_fit_wf(
        bold_file=bold_file,
        metadata=bold_metadata,
        mem_gb=mem_gb,
    )
    rapidtide_wf.inputs.inputnode.boldref = functional_cache['boldref']
    rapidtide_wf.inputs.inputnode.confounds = functional_cache['confounds']
    rapidtide_wf.inputs.inputnode.skip_vols = skip_vols

    workflow.connect([
        (dseg_to_boldref, rapidtide_wf, [('output_image', 'inputnode.dseg')]),
        (boldref_buffer, rapidtide_wf, [
            ('bold', 'inputnode.bold'),
            ('bold_mask', 'inputnode.bold_mask'),
        ]),
        (rapidtide_wf, outputnode, [
            ('outputnode.rapidtide_root', 'rapidtide_root'),
            ('outputnode.delay_map', 'delay_map'),
            ('outputnode.lagtcgenerator', 'lagtcgenerator'),
            ('outputnode.valid_mask', 'valid_mask'),
            ('outputnode.runoptions', 'runoptions'),
        ]),
    ])  # fmt:skip

    return clean_datasinks(workflow, bold_file=bold_file)


def init_denoise_single_run_wf(*, bold_file: str):
    """Denoise a single run using rapidtide.

    Parameters
    ----------
    bold_file : str
        BOLD file used as name source for datasinks.

    Inputs
    ------
    bold
    bold_mask
    rapidtide_root
    lagtcgenerator
    delay_map
    skip_vols
    valid_mask
    runoptions
    anat_dseg
    boldref2anat
    anat2outputspaces
    """

    from fmriprep.utils.misc import estimate_bold_mem_usage
    from nipype.interfaces import utility as niu
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.utility import KeySelect
    from smriprep.interfaces.templateflow import TemplateFlowSelect

    from fmripost_rapidtide.interfaces.bids import DerivativesDataSink
    from fmripost_rapidtide.interfaces.misc import ApplyTransforms
    from fmripost_rapidtide.interfaces.rapidtide import RetroRegress
    from fmripost_rapidtide.utils.utils import load_json
    from fmripost_rapidtide.workflows.confounds import init_denoising_confounds_wf
    from fmripost_rapidtide.workflows.rapidtide import init_rapidtide_confounds_wf

    mem_gb = estimate_bold_mem_usage(bold_file)[1]

    workflow = Workflow(name=_get_wf_name(bold_file, 'rapidtide_denoise'))
    workflow.__postdesc__ = """\
Identification and removal of traveling wave artifacts was performed using rapidtide.
"""

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'bold',
                'bold_mask',
                'rapidtide_root',
                'lagtcgenerator',
                'delay_map',
                'skip_vols',
                'valid_mask',
                'runoptions',
                # transforms to output spaces
                'anat_dseg',
                'boldref2anat',
                'anat2outputspaces',
                'templates',
            ],
        ),
        name='inputnode',
    )

    denoise_bold = pe.Node(
        RetroRegress(),
        name='denoise_bold',
    )
    workflow.connect([
        (inputnode, denoise_bold, [
            ('bold', 'in_file'),
            ('rapidtide_root', 'datafileroot'),
        ]),
    ])  # fmt:skip

    # TODO: Warp denoised data to target spaces
    # Now we'd set up a template iterator workflow.
    # We just need to apply the boldref2anat and anat2outputspaces transforms here,
    # since denoising was done in boldref space.
    spaces = config.workflow.spaces
    nonstd_spaces = set(spaces.get_nonstandard())

    boldref_out = bool(nonstd_spaces.intersection(('func', 'run', 'bold', 'boldref', 'sbref')))
    if boldref_out:
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
            (denoise_bold, ds_denoised_bold, [
                ('denoised', 'in_file'),
                (('denoised_json', load_json), 'meta_dict'),
            ]),
        ])  # fmt:skip

    if nonstd_spaces.intersection(('anat', 'T1w')):
        # Warp denoised data to anatomical space
        denoised_to_anat = pe.Node(
            ApplyTransforms(
                input_image_type=3,
                interpolation='LanczosWindowedSinc',
                args='--verbose',
            ),
            name='denoised_to_anat',
        )
        workflow.connect([
            (inputnode, denoised_to_anat, [
                ('anat_dseg', 'reference_image'),
                ('boldref2anat', 'transforms'),
            ]),
            (denoise_bold, denoised_to_anat, [('denoised', 'input_image')]),
        ])  # fmt:skip

        ds_denoised_bold_anat = pe.Node(
            DerivativesDataSink(
                compress=True,
                space='T1w',
                desc='denoised',
                suffix='bold',
            ),
            name='ds_denoised_bold_anat',
            run_without_submitting=True,
        )
        workflow.connect([
            (denoise_bold, ds_denoised_bold_anat, [(('denoised_json', load_json), 'meta_dict')]),
            (denoised_to_anat, ds_denoised_bold_anat, [('output_image', 'in_file')]),
        ])  # fmt:skip

    if spaces.cached.get_spaces(nonstandard=False, dim=(3,)):
        # Warp denoised data to template spaces
        for space in spaces.cached.get_spaces(nonstandard=False, dim=(3,)):
            select_xfm = pe.Node(
                KeySelect(fields=['anat2outputspaces'], key=space.fullname),
                name=f'select_xfm_{space.fullname}',
                run_without_submitting=True,
            )
            select_tpl = pe.Node(
                TemplateFlowSelect(template=space.fullname, resolution=space.spec['res']),
                name=f'select_tpl_{space.fullname}',
            )
            workflow.connect([
                (inputnode, select_xfm, [
                    ('anat2outputspaces', 'anat2outputspaces'),
                    ('templates', 'keys'),
                ]),
            ])  # fmt:skip

            merge_xfms = pe.Node(
                niu.Merge(2),
                name=f'merge_xfms_{space.fullname}',
            )
            workflow.connect([
                (inputnode, merge_xfms, [('boldref2anat', 'in2')]),
                (select_xfm, merge_xfms, [('anat2outputspaces', 'in1')]),
            ])  # fmt:skip

            # Warp BOLD image to MNI152NLin6Asym
            warp_denoised_to_template = pe.Node(
                ApplyTransforms(
                    input_image_type=3,
                    interpolation='LanczosWindowedSinc',
                    args='--verbose',
                ),
                name=f'warp_denoised_to_{space.fullname}',
            )
            workflow.connect([
                (denoise_bold, warp_denoised_to_template, [('denoised', 'input_image')]),
                (merge_xfms, warp_denoised_to_template, [('out', 'transforms')]),
                (select_tpl, warp_denoised_to_template, [('brain_mask', 'reference_image')]),
            ])  # fmt:skip

            ds_denoised_bold_template = pe.Node(
                DerivativesDataSink(
                    compress=True,
                    desc='denoised',
                    suffix='bold',
                ),
                name=f'ds_denoised_bold_{space.fullname}',
                run_without_submitting=True,
            )
            # TODO: Pass in space, resolution, and cohort
            workflow.connect([
                (denoise_bold, ds_denoised_bold_template, [
                    (('denoised_json', load_json), 'meta_dict'),
                ]),
                (warp_denoised_to_template, ds_denoised_bold_template, [
                    ('output_image', 'in_file'),
                ]),
            ])  # fmt:skip

    # Generate voxel-wise regressors file(s)
    # TODO: Warp delay map to target spaces and generate voxelwise regressor files from those
    rapidtide_confounds_wf = init_rapidtide_confounds_wf(
        bold_file=bold_file,
        metadata={},
        mem_gb=mem_gb,
    )
    workflow.connect([
        (inputnode, rapidtide_confounds_wf, [
            ('bold', 'inputnode.bold'),
            ('bold_mask', 'inputnode.bold_mask'),
            # Inputs to warp to target spaces
            ('anat_dseg', 'inputnode.anat_dseg'),
            ('boldref2anat', 'inputnode.boldref2anat'),
            ('anat2outputspaces', 'inputnode.anat2outputspaces'),
            # Rapidtide outputs
            ('lagtcgenerator', 'inputnode.lagtcgenerator'),
            ('delay_map', 'inputnode.delay_map'),
            ('valid_mask', 'inputnode.valid_mask'),
            ('skip_vols', 'inputnode.skip_vols'),
        ]),
    ])  # fmt:skip

    # Generate non-rapidtide confounds (e.g., FC inflation metric)
    denoising_confounds_wf = init_denoising_confounds_wf(bold_file=bold_file, mem_gb=mem_gb)
    workflow.connect([
        (inputnode, denoising_confounds_wf, [
            ('bold', 'inputnode.preprocessed_bold'),
            ('bold_mask', 'inputnode.mask'),
            ('templates', 'inputnode.templates'),
            ('anat2outputspaces', 'inputnode.anat2outputspaces'),
            ('boldref2anat', 'inputnode.boldref2anat'),
        ]),
        (denoise_bold, denoising_confounds_wf, [('denoised', 'inputnode.denoised_bold')]),
    ])  # fmt:skip

    return clean_datasinks(workflow, bold_file=bold_file)


def clean_datasinks(workflow: pe.Workflow, bold_file: str) -> pe.Workflow:
    """Overwrite attributes of DataSinks."""
    for node in workflow.list_node_names():
        node_name = node.split('.')[-1]
        if node_name.startswith('ds_'):
            workflow.get_node(node).interface.out_path_base = ''
            workflow.get_node(node).inputs.base_directory = str(config.execution.output_dir)
            workflow.get_node(node).inputs.source_file = bold_file

        if node_name.startswith('ds_report_'):
            workflow.get_node(node).inputs.datatype = 'figures'

    return workflow


def get_nss(confounds_file):
    """Get number of non-steady state volumes."""
    import numpy as np
    import pandas as pd

    df = pd.read_table(confounds_file)

    nss_cols = [c for c in df.columns if c.startswith('non_steady_state_outlier')]

    dummy_scans = 0
    if nss_cols:
        initial_volumes_df = df[nss_cols]
        dummy_scans = np.any(initial_volumes_df.to_numpy(), axis=1)
        dummy_scans = np.where(dummy_scans)[0]

        # reasonably assumes all NSS volumes are contiguous
        dummy_scans = int(dummy_scans[-1] + 1)

    return dummy_scans
