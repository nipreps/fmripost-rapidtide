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
from fmripost_rapidtide.interfaces.bids import DerivativesDataSink
from fmripost_rapidtide.interfaces.rapidtide import Rapidtide
from fmripost_rapidtide.utils.utils import _get_wf_name


def init_rapidtide_wf(
    *,
    bold_file: str,
    metadata: dict,
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
    bold_std
        BOLD series in template space
    bold_mask_std
        BOLD series mask in template space
    dseg_std
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

    workflow = Workflow(name=_get_wf_name(bold_file, 'rapidtide'))
    workflow.__postdesc__ = """\
Automatic removal of motion artifacts using independent component analysis
[Rapidtide, @rapidtide] was performed on the *preprocessed BOLD on MNI152NLin6Asym space*.
"""

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'bold_std',
                'bold_mask_std',
                'dseg_std',
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
                'denoised',
            ],
        ),
        name='outputnode',
    )

    rm_non_steady_state = pe.Node(
        niu.Function(function=_remove_volumes, output_names=['bold_cut']),
        name='rm_nonsteady',
    )
    workflow.connect([
        (inputnode, rm_non_steady_state, [
            ('skip_vols', 'skip_vols'),
            ('bold_std', 'bold_file'),
        ]),
    ])  # fmt:skip

    # Split tissue-type segmentation to get GM and WM masks
    split_tissues = pe.Node(
        niu.IdentityInterface(
            fields=['dseg'],
        ),
        name='split_tissues',
    )
    workflow.connect([(inputnode, split_tissues, [('dseg_std', 'dseg')])])

    # Run the Rapidtide classifier
    rapidtide = pe.Node(
        Rapidtide(
            denoising=True,
            datatstep=metadata['RepetitionTime'],
            autosync=config.workflow.autosync,
            filterband=config.workflow.filterband,
            passvec=config.workflow.passvec,
            stopvec=config.workflow.stopvec,
            numestreps=config.workflow.numestreps,
            detrendorder=config.workflow.detrendorder,
            gausssigma=config.workflow.gausssigma,
            confoundfilespec=config.workflow.confoundfilespec,
            confound_power=config.workflow.confound_power,
            confound_deriv=config.workflow.confound_deriv,
            globalsignalmethod=config.workflow.globalsignalmethod,
            globalpcacomponents=config.workflow.globalpcacomponents,
            numtozero=config.workflow.numtozero,
            timerange=config.workflow.timerange,
            corrweighting=config.workflow.corrweighting,
            simcalcrange=config.workflow.simcalcrange,
            fixeddelayvalue=config.workflow.fixeddelayvalue,
            lag_extrema=config.workflow.lag_extrema,
            widthmax=config.workflow.widthmax,
            bipolar=config.workflow.bipolar,
            lagminthresh=config.workflow.lagminthresh,
            lagmaxthresh=config.workflow.lagmaxthresh,
            ampthresh=config.workflow.ampthresh,
            sigmathresh=config.workflow.sigmathresh,
            pcacomponents=config.workflow.pcacomponents,
            convergencethresh=config.workflow.convergencethresh,
            maxpasses=config.workflow.maxpasses,
            glmsourcefile=config.workflow.glmsourcefile,
            glmderivs=config.workflow.glmderivs,
            outputlevel=config.workflow.outputlevel,
            territorymap=config.workflow.territorymap,
            respdelete=config.workflow.respdelete,
        ),
        name='rapidtide',
    )
    workflow.connect([
        (inputnode, rapidtide, [
            ('bold_std', 'in_file'),
            ('brain_mask_std', 'brainmask'),
            ('confounds', 'motpars'),
            ('skip_vols', 'skip_vols'),
        ]),
        (split_tissues, rapidtide, [
            ('dseg', 'graymattermask'),
            ('dseg', 'whitemattermask'),
            ('dseg', 'globalmeaninclude'),  # GM mask for initial regressor selection
            ('dseg', 'refineinclude'),  # GM mask for refinement
            ('dseg', 'offsetinclude'),  # GM mask for offset calculation
        ]),
        (rapidtide, outputnode, [
            ('delay_map', 'delay_map'),
            ('regressor_file', 'regressor_file'),
            ('denoised', 'denoised'),
        ])
    ])  # fmt:skip

    # Generate figures for report

    return workflow


def _remove_volumes(bold_file, skip_vols):
    """Remove skip_vols from bold_file."""
    import os

    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    if skip_vols == 0:
        return bold_file

    out = fname_presuffix(bold_file, prefix='cut_', newpath=os.getcwd())
    bold_img = nb.load(bold_file)
    bold_img.__class__(
        bold_img.dataobj[..., skip_vols:], bold_img.affine, bold_img.header
    ).to_filename(out)
    return out


def _add_volumes(bold_file, bold_cut_file, skip_vols):
    """Prepend skip_vols from bold_file onto bold_cut_file."""
    import os

    import nibabel as nb
    import numpy as np
    from nipype.utils.filemanip import fname_presuffix

    if skip_vols == 0:
        return bold_cut_file

    bold_img = nb.load(bold_file)
    bold_cut_img = nb.load(bold_cut_file)

    bold_data = np.concatenate((bold_img.dataobj[..., :skip_vols], bold_cut_img.dataobj), axis=3)

    out = fname_presuffix(bold_cut_file, prefix='addnonsteady_', newpath=os.getcwd())
    bold_img.__class__(bold_data, bold_img.affine, bold_img.header).to_filename(out)
    return out
