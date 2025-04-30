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
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""
Calculate BOLD confounds
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_carpetplot_wf

"""


def init_denoising_confounds_wf(
    bold_file: str,
    mem_gb: float,
    name: str = 'denoising_confounds_wf',
):
    from nipype.interfaces import utility as niu
    from nipype.pipeline import engine as pe
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    from fmripost_rapidtide.config import DEFAULT_MEMORY_MIN_GB
    from fmripost_rapidtide.interfaces.bids import DerivativesDataSink
    from fmripost_rapidtide.interfaces.confounds import FCInflation
    from fmripost_rapidtide.interfaces.reportlets import FCInflationPlotRPT

    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'preprocessed_bold',
                'denoised_bold',
                'rapidtide_bold',
                'mask',
            ],
        ),
        name='inputnode',
    )

    # Prepare to merge FC inflation results
    merge_fci_confounds = pe.Node(
        niu.Merge(3),
        name='merge_fci_confounds',
        run_without_submitting=True,
    )
    merge_fci_metrics = pe.Node(
        niu.Merge(3),
        name='merge_fci_metrics',
        run_without_submitting=True,
    )

    # Calculate FC inflation for each BOLD type
    bold_types = ['preprocessed', 'denoised', 'rapidtide']
    for i_type, bold_type in enumerate(bold_types):
        fc_inflation = pe.Node(
            FCInflation(),
            name=f'fc_inflation_{bold_type}',
            mem_gb=mem_gb,
        )
        workflow.connect([
            (inputnode, fc_inflation, [
                (f'{bold_type}_bold', 'in_file'),
                ('mask', 'mask'),
            ]),
            (fc_inflation, merge_fci_confounds, [('fc_inflation', f'in{i_type + 1}')]),
            (fc_inflation, merge_fci_metrics, [('metrics', f'in{i_type + 1}')]),
        ])  # fmt:skip

    # Combine the FC inflation results
    merge_fci = pe.Node(
        niu.Function(
            input_names=['confounds', 'metrics', 'prefixes'],
            function=_merge_fci,
        ),
        name='merge_fci',
        run_without_submitting=True,
    )
    merge_fci.inputs.prefixes = bold_types
    workflow.connect([
        (merge_fci_confounds, merge_fci, [('out', 'confounds')]),
        (merge_fci_metrics, merge_fci, [('out', 'metrics')]),
    ])  # fmt:skip

    ds_confounds = pe.Node(
        DerivativesDataSink(
            source_file=bold_file,
            dismiss_entities=('echo', 'den', 'res', 'space'),
            datatype='func',
            desc='confounds',
            suffix='timeseries',
            extension='tsv',
        ),
        name='ds_confounds',
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )
    workflow.connect([(merge_fci, ds_confounds, [('confounds_file', 'in_file')])])

    ds_metrics = pe.Node(
        DerivativesDataSink(
            source_file=bold_file,
            dismiss_entities=('echo', 'den', 'res', 'space'),
            datatype='func',
            desc='confounds',
            suffix='metrics',
            extension='json',
        ),
        name='ds_metrics',
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )
    workflow.connect([(merge_fci, ds_metrics, [('confounds_metrics', 'in_file')])])

    # Generate reportlets
    plot_fcinflation = pe.Node(
        FCInflationPlotRPT(),
        name='plot_fcinflation',
    )
    workflow.connect([(merge_fci, plot_fcinflation, [('confounds_file', 'fcinflation_file')])])

    ds_report_fcinflation = pe.Node(
        DerivativesDataSink(
            desc='fcinflation',
            suffix='bold',
            extension='.svg',
        ),
        name='ds_report_fcinflation',
        run_without_submitting=True,
    )
    workflow.connect([(plot_fcinflation, ds_report_fcinflation, [('out_report', 'in_file')])])

    return workflow


def _merge_fci(confounds, metrics, prefixes):
    """Merge FC inflation results."""
    import os

    import pandas as pd

    confounds_dfs = []
    metrics = {}
    for i_prefix, prefix in enumerate(prefixes):
        # Add prefix to column names
        prefix_confounds_file = confounds[i_prefix]
        prefix_confounds_df = pd.read_table(prefix_confounds_file)
        prefix_confounds_df.columns = [f'{prefix}_{col}' for col in prefix_confounds_df.columns]
        confounds_dfs.append(prefix_confounds_df)

        prefix_metrics = metrics[i_prefix]
        prefix_metrics = {f'{prefix}_{key}': value for key, value in prefix_metrics.items()}
        metrics.update(prefix_metrics)

    merged_confounds_df = pd.concat(confounds_dfs, axis=1)
    merged_confounds_file = os.path.abspath('confounds.tsv')
    merged_confounds_df.to_csv(merged_confounds_file, sep='\t', index=False)
    return merged_confounds_file, metrics


def init_carpetplot_wf(
    mem_gb: float,
    metadata: dict,
    cifti_output: bool,
    name: str = 'bold_carpet_wf',
):
    """Build a workflow to generate *carpet* plots.

    Resamples the MNI parcellation
    (ad-hoc parcellation derived from the Harvard-Oxford template and others).

    Parameters
    ----------
    mem_gb : :obj:`float`
        Size of BOLD file in GB - please note that this size
        should be calculated after resamplings that may extend
        the FoV
    metadata : :obj:`dict`
        BIDS metadata for BOLD file
    name : :obj:`str`
        Name of workflow (default: ``bold_carpet_wf``)

    Inputs
    ------
    bold
        BOLD image, after the prescribed corrections (STC, HMC and SDC)
        when available.
    bold_mask
        BOLD series mask
    confounds_file
        TSV of all aggregated confounds
    boldref2anat_xfm
        Affine matrix that maps the BOLD reference space into alignment with
        the anatomical (T1w) space
    std2anat_xfm
        ANTs-compatible affine-and-warp transform file
    cifti_bold
        BOLD image in CIFTI format, to be used in place of volumetric BOLD
    crown_mask
        Mask of brain edge voxels
    acompcor_mask
        Mask of deep WM+CSF
    dummy_scans
        Number of nonsteady states to be dropped at the beginning of the timeseries.

    Outputs
    -------
    out_carpetplot
        Path of the generated SVG file
    """
    from nipype.interfaces import utility as niu
    from nipype.pipeline import engine as pe
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from templateflow.api import get as get_template

    from fmripost_rapidtide.config import DEFAULT_MEMORY_MIN_GB
    from fmripost_rapidtide.interfaces import DerivativesDataSink
    from fmripost_rapidtide.interfaces.confounds import FMRISummary
    from fmripost_rapidtide.interfaces.misc import ApplyTransforms

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'bold',
                'bold_mask',
                'confounds_file',
                'boldref2anat_xfm',
                'std2anat_xfm',
                'cifti_bold',
                'crown_mask',
                'acompcor_mask',
                'dummy_scans',
            ]
        ),
        name='inputnode',
    )

    outputnode = pe.Node(niu.IdentityInterface(fields=['out_carpetplot']), name='outputnode')

    # Carpetplot and confounds plot
    conf_plot = pe.Node(
        FMRISummary(
            tr=metadata['RepetitionTime'],
            confounds_list=[
                ('global_signal', None, 'GS'),
                ('csf', None, 'CSF'),
                ('white_matter', None, 'WM'),
                ('std_dvars', None, 'DVARS'),
                ('framewise_displacement', 'mm', 'FD'),
            ],
        ),
        name='conf_plot',
        mem_gb=mem_gb,
    )
    ds_report_bold_conf = pe.Node(
        DerivativesDataSink(
            desc='carpetplot',
            datatype='figures',
            extension='svg',
            dismiss_entities=('echo', 'den', 'res'),
        ),
        name='ds_report_bold_conf',
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    parcels = pe.Node(niu.Function(function=_carpet_parcellation), name='parcels')
    parcels.inputs.nifti = not cifti_output
    # List transforms
    mrg_xfms = pe.Node(niu.Merge(2), name='mrg_xfms')

    # Warp segmentation into EPI space
    resample_parc = pe.Node(
        ApplyTransforms(
            dimension=3,
            input_image=str(
                get_template(
                    'MNI152NLin2009cAsym',
                    resolution=1,
                    desc='carpet',
                    suffix='dseg',
                    extension=['.nii', '.nii.gz'],
                )
            ),
            invert_transform_flags=[True, False],
            interpolation='GenericLabel',
            args='-u int',
        ),
        name='resample_parc',
    )

    workflow = Workflow(name=name)
    if cifti_output:
        workflow.connect(inputnode, 'cifti_bold', conf_plot, 'in_cifti')

    workflow.connect([
        (inputnode, mrg_xfms, [
            ('boldref2anat_xfm', 'in1'),
            ('std2anat_xfm', 'in2'),
        ]),
        (inputnode, resample_parc, [('bold_mask', 'reference_image')]),
        (inputnode, parcels, [('crown_mask', 'crown_mask')]),
        (inputnode, parcels, [('acompcor_mask', 'acompcor_mask')]),
        (inputnode, conf_plot, [
            ('bold', 'in_nifti'),
            ('confounds_file', 'confounds_file'),
            ('dummy_scans', 'drop_trs'),
        ]),
        (mrg_xfms, resample_parc, [('out', 'transforms')]),
        (resample_parc, parcels, [('output_image', 'segmentation')]),
        (parcels, conf_plot, [('out', 'in_segm')]),
        (conf_plot, ds_report_bold_conf, [('out_file', 'in_file')]),
        (conf_plot, outputnode, [('out_file', 'out_carpetplot')]),
    ])  # fmt:skip
    return workflow


def _carpet_parcellation(segmentation, crown_mask, acompcor_mask, nifti=False):
    """Generate the union of two masks."""
    from pathlib import Path

    import nibabel as nb
    import numpy as np

    img = nb.load(segmentation)

    lut = np.zeros((256,), dtype='uint8')
    lut[100:201] = 1 if nifti else 0  # Ctx GM
    lut[30:99] = 2 if nifti else 0  # dGM
    lut[1:11] = 3 if nifti else 1  # WM+CSF
    lut[255] = 5 if nifti else 0  # Cerebellum
    # Apply lookup table
    seg = lut[np.uint16(img.dataobj)]
    seg[np.bool_(nb.load(crown_mask).dataobj)] = 6 if nifti else 2
    # Separate deep from shallow WM+CSF
    seg[np.bool_(nb.load(acompcor_mask).dataobj)] = 4 if nifti else 1

    outimg = img.__class__(seg.astype('uint8'), img.affine, img.header)
    outimg.set_data_dtype('uint8')
    out_file = Path('segments.nii.gz').absolute()
    outimg.to_filename(out_file)
    return str(out_file)
