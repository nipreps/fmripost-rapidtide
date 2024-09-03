"""Nilearn interfaces."""

import os

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
)
from nipype.interfaces.nilearn import NilearnBaseInterface


class _MeanImageInputSpec(BaseInterfaceInputSpec):
    bold_file = File(
        exists=True,
        mandatory=True,
        desc='A 4D BOLD file to process.',
    )
    mask_file = File(
        exists=True,
        mandatory=False,
        desc='A binary brain mask.',
    )
    out_file = File(
        'mean.nii.gz',
        usedefault=True,
        exists=False,
        desc='The name of the mean file to write out. mean.nii.gz by default.',
    )


class _MeanImageOutputSpec(TraitedSpec):
    out_file = File(
        exists=True,
        desc='Mean output file.',
    )


class MeanImage(NilearnBaseInterface, SimpleInterface):
    """MeanImage images."""

    input_spec = _MeanImageInputSpec
    output_spec = _MeanImageOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nb
        from nilearn.masking import apply_mask, unmask
        from nipype.interfaces.base import isdefined

        if isdefined(self.inputs.mask_file):
            data = apply_mask(self.inputs.bold_file, self.inputs.mask_file)
            mean_data = data.mean(axis=0)
            mean_img = unmask(mean_data, self.inputs.mask_file)
        else:
            in_img = nb.load(self.inputs.bold_file)
            mean_data = in_img.get_fdata().mean(axis=3)
            mean_img = nb.Nifti1Image(mean_data, in_img.affine, in_img.header)

        self._results['out_file'] = os.path.join(runtime.cwd, self.inputs.out_file)
        mean_img.to_filename(self._results['out_file'])

        return runtime


class _SplitDsegInputSpec(BaseInterfaceInputSpec):
    dseg = File(
        exists=True,
        mandatory=True,
        desc='A tissue type dseg file to split up.',
    )


class _SplitDsegOutputSpec(TraitedSpec):
    gm = File(
        exists=True,
        mandatory=True,
        desc='Gray matter tissue type mask.',
    )
    wm = File(
        exists=True,
        mandatory=True,
        desc='White matter tissue type mask.',
    )
    csf = File(
        exists=True,
        mandatory=True,
        desc='Cerebrospinal fluid tissue type mask.',
    )


class SplitDseg(NilearnBaseInterface, SimpleInterface):
    """Split up a dseg into tissue type-wise masks."""

    input_spec = _SplitDsegInputSpec
    output_spec = _SplitDsegOutputSpec

    def _run_interface(self, runtime):
        import os

        from nilearn.image import math_img

        gm_img = math_img('img == 1', img=self.inputs.dseg)
        self._results['gm'] = os.path.abspath('gm.nii.gz')
        gm_img.to_filename(self._results['gm'])

        wm_img = math_img('img == 2', img=self.inputs.dseg)
        self._results['wm'] = os.path.abspath('wm.nii.gz')
        wm_img.to_filename(self._results['wm'])

        csf_img = math_img('img == 3', img=self.inputs.dseg)
        self._results['csf'] = os.path.abspath('csf.nii.gz')
        csf_img.to_filename(self._results['csf'])

        return runtime
