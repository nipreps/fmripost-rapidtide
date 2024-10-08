"""Interfaces to run rapidtide."""

import os

from nipype.interfaces.base import (
    CommandLine,
    CommandLineInputSpec,
    File,
    TraitedSpec,
    traits,
)


class _RapidtideInputSpec(CommandLineInputSpec):
    in_file = File(
        exists=True,
        argstr='%s',
        position=-2,
        mandatory=True,
        desc='File to denoise',
    )
    outputname = traits.Str(
        argstr='%s',
        position=-1,
        mandatory=True,
        desc='Output name',
    )
    # Set by the workflow
    denoising = traits.Bool(
        argstr='--denoising',
        mandatory=False,
    )
    brainmask = File(
        exists=True,
        argstr='--brainmask %s',
        mandatory=False,
    )
    graymattermask = File(
        exists=True,
        argstr='--graymattermask %s',
        mandatory=False,
    )
    whitemattermask = File(
        exists=True,
        argstr='--whitemattermask %s',
        mandatory=False,
    )
    datatstep = traits.Float(
        argstr='--datatstep %f',
        mandatory=False,
    )
    padseconds = traits.Float(
        argstr='--padseconds %f',
        mandatory=False,
    )
    globalmeaninclude = File(
        exists=True,
        argstr='--globalmeaninclude %s',
        mandatory=False,
    )
    motionfile = File(
        exists=True,
        argstr='--motionfile %s',
        mandatory=False,
    )
    corrmask = File(
        exists=True,
        argstr='--corrmask %s',
        mandatory=False,
    )
    refineinclude = File(
        exists=True,
        argstr='--refineinclude %s',
        mandatory=False,
    )
    offsetinclude = File(
        exists=True,
        argstr='--offsetinclude %s',
        mandatory=False,
    )
    # Set by the user
    autosync = traits.Bool(
        argstr='--autosync',
        mandatory=False,
    )
    filterband = traits.Enum(
        'vlf',
        'lfo',
        'resp',
        'cardiac',
        'hrv_ulf',
        'hrv_vlf',
        'hrv_lf',
        'hrv_hf',
        'hrv_vhf',
        'lfo_legacy',
        argstr='--filterband %s',
        mandatory=False,
    )
    filterfreqs = traits.List(
        traits.Float,
        argstr='--filterfreqs %s',
        mandatory=False,
        minlen=2,
        maxlen=2,
    )
    filterstopfreqs = traits.List(
        traits.Float,
        argstr='--filterstopfreqs %s',
        mandatory=False,
        minlen=2,
        maxlen=2,
    )
    numnull = traits.Int(
        argstr='--numnull %d',
        mandatory=False,
    )
    detrendorder = traits.Int(
        argstr='--detrendorder %d',
        mandatory=False,
    )
    spatialfilt = traits.Float(
        argstr='--spatialfilt %f',
        mandatory=False,
    )
    confoundfile = File(
        exists=True,
        argstr='--confoundfile %s',
        mandatory=False,
    )
    confoundpowers = traits.Int(
        argstr='--confoundpowers %d',
        mandatory=False,
    )
    confoundderiv = traits.Bool(
        argstr='--confoundderiv',
        mandatory=False,
    )
    globalsignalmethod = traits.Enum(
        'sum',
        'meanscale',
        'pca',
        'random',
        argstr='--globalsignalmethod %s',
        mandatory=False,
    )
    globalpcacomponents = traits.Float(
        argstr='--globalpcacomponents %f',
        mandatory=False,
    )
    numskip = traits.Int(
        argstr='--numskip %d',
        mandatory=False,
    )
    numtozero = traits.Int(
        argstr='--numtozero %d',
        mandatory=False,
    )
    timerange = traits.List(
        traits.Int,
        argstr='--timerange %s',
        mandatory=False,
        minlen=2,
        maxlen=2,
    )
    corrweighting = traits.Enum(
        'phat',
        'liang',
        'eckart',
        'regressor',
        argstr='--corrweighting %s',
        mandatory=False,
    )
    simcalcrange = traits.List(
        traits.Int,
        argstr='--simcalcrange %s',
        mandatory=False,
        minlen=2,
        maxlen=2,
    )
    fixdelay = traits.Float(
        argstr='--fixdelay %f',
        mandatory=False,
    )
    searchrange = traits.List(
        traits.Int,
        argstr='--searchrange %s',
        mandatory=False,
        minlen=2,
        maxlen=2,
    )
    sigmalimit = traits.Float(
        argstr='--sigmalimit %f',
        mandatory=False,
    )
    bipolar = traits.Bool(
        argstr='--bipolar',
        mandatory=False,
    )
    lagminthresh = traits.Float(
        argstr='--lagminthresh %f',
        mandatory=False,
    )
    lagmaxthresh = traits.Float(
        argstr='--lagmaxthresh %f',
        mandatory=False,
    )
    ampthresh = traits.Float(
        argstr='--ampthresh %f',
        mandatory=False,
    )
    sigmathresh = traits.Float(
        argstr='--sigmathresh %f',
        mandatory=False,
    )
    pcacomponents = traits.Float(
        argstr='--pcacomponents %f',
        mandatory=False,
    )
    convergencethresh = traits.Float(
        argstr='--convergencethresh %f',
        mandatory=False,
    )
    maxpasses = traits.Int(
        argstr='--maxpasses %d',
        mandatory=False,
    )
    glmsourcefile = File(
        exists=True,
        argstr='--glmsourcefile %s',
        mandatory=False,
    )
    glmderivs = traits.Int(
        argstr='--glmderivs %d',
        mandatory=False,
    )
    outputlevel = traits.Enum(
        'min',
        'less',
        'normal',
        'more',
        'max',
        argstr='--outputlevel %s',
        mandatory=False,
    )
    territorymap = File(
        exists=True,
        argstr='--territorymap %s',
        mandatory=False,
    )
    autorespdelete = traits.Bool(
        argstr='--autorespdelete',
        mandatory=False,
    )
    nprocs = traits.Int(
        default=1,
        usedefault=True,
        argstr='--nprocs %d',
        mandatory=False,
    )


class _RapidtideOutputSpec(TraitedSpec):
    delay_map = File(exists=True, desc='3D map of optimal delay times')
    regressor_file = File(exists=True, desc='Time series of refined regressor')
    denoised = File(exists=True, desc='Denoised time series')


class Rapidtide(CommandLine):
    """Run the rapidtide command-line interface."""

    _cmd = 'rapidtide --noprogressbar'
    input_spec = _RapidtideInputSpec
    output_spec = _RapidtideOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        out_dir = os.getcwd()
        outputname = self.inputs.outputname
        outputs['delay_map'] = os.path.join(out_dir, f'{outputname}_desc-maxtime_map.nii.gz')
        outputs['regressor_file'] = os.path.join(
            out_dir,
            f'{outputname}_desc-refinedmovingregressor_timeseries.tsv.gz',
        )
        outputs['denoised'] = os.path.join(
            out_dir,
            f'{outputname}_desc-lfofilterCleaned_bold.nii.gz',
        )
        return outputs
