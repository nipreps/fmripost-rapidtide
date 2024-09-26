"""Interfaces to run rapidtide."""

import os

import yaml
from nipype.interfaces.base import (
    CommandLine,
    CommandLineInputSpec,
    DynamicTraitedSpec,
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
    prefix = traits.Str(
        argstr='%s',
        position=-1,
        mandatory=False,
        genfile=True,
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
    prefix = traits.Str(desc='Directory containing the results, with prefix.')
    lagtimesfile = File(exists=True, desc='3D map of optimal delay times')
    lagtcgeneratorfile = File(exists=True, desc='Time series of refined regressor')
    maskfile = File(exists=True, desc='Mask file (usually called XXX_desc-corrfit_mask.nii.gz)')


class Rapidtide(CommandLine):
    """Run the rapidtide command-line interface."""

    _cmd = 'rapidtide --noprogressbar --spcalculation --noglm'
    input_spec = _RapidtideInputSpec
    output_spec = _RapidtideOutputSpec

    def _gen_filename(self, name):
        if name == 'prefix':
            return os.path.join(os.getcwd(), 'rapidtide')

        return None

    def _list_outputs(self):
        outputs = self._outputs().get()
        prefix = self.inputs.prefix
        outputs['prefix'] = prefix
        outputs['lagtimesfile'] = f'{prefix}_desc-maxtime_map.nii.gz'
        outputs['lagtcgeneratorfile'] = f'{prefix}_desc-lagtcgenerator_timeseries.tsv.gz'
        outputs['maskfile'] = f'{prefix}_desc-corrfit_mask.nii.gz'

        return outputs


class _RetroLagTCSInputSpec(CommandLineInputSpec):
    in_file = File(
        exists=True,
        argstr='%s',
        position=0,
        mandatory=True,
        desc='The name of 4D nifti fmri target file.',
    )
    maskfile = File(
        exists=True,
        argstr='%s',
        position=1,
        mandatory=True,
        desc='The mask file to use (usually called XXX_desc-corrfit_mask.nii.gz)',
    )
    lagtimesfile = File(
        exists=True,
        argstr='%s',
        position=2,
        mandatory=True,
        desc='The name of the lag times file (usually called XXX_desc-maxtime_map.nii.gz)',
    )
    lagtcgeneratorfile = File(
        exists=True,
        argstr='%s',
        position=3,
        mandatory=True,
        desc=(
            'The root name of the lagtc generator file '
            '(usually called XXX_desc-lagtcgenerator_timeseries)'
        ),
    )
    prefix = traits.Str(
        argstr='%s',
        position=4,
        mandatory=False,
        genfile=True,
        desc='Output root.',
    )
    glmderivs = traits.Int(
        argstr='--glmderivs %d',
        mandatory=False,
        desc='When doing final GLM, include derivatives up to NDERIVS order. Default is 0.',
        default=0,
        usedefault=True,
    )
    nprocs = traits.Int(
        default=1,
        usedefault=True,
        argstr='--nprocs %d',
        mandatory=False,
        desc=(
            'Use NPROCS worker processes for multiprocessing. '
            'Setting NPROCS to less than 1 sets the number of worker processes to n_cpus.'
        ),
    )
    numskip = traits.Int(
        argstr='--numskip %d',
        default=0,
        usedefault=True,
        mandatory=False,
        desc='Skip NUMSKIP points at the beginning of the fmri file.',
    )
    noprogressbar = traits.Bool(
        argstr='--noprogressbar',
        mandatory=False,
        default=True,
        usedefault=True,
        desc='Will disable showing progress bars (helpful if stdout is going to a file).',
    )
    debug = traits.Bool(
        argstr='--debug',
        mandatory=False,
        default=False,
        usedefault=True,
        desc='Output lots of helpful information.',
    )


class _RetroLagTCSOutputSpec(DynamicTraitedSpec):
    filter_file = File(exists=True, desc='Filter file')


class RetroLagTCS(CommandLine):
    """Run the retrolagtcs command-line interface."""

    _cmd = 'retrolagtcs'
    input_spec = _RetroLagTCSInputSpec
    output_spec = _RetroLagTCSOutputSpec

    def _gen_filename(self, name):
        if name == 'prefix':
            return os.path.join(os.getcwd(), 'retrolagtcs')

        return None

    def _list_outputs(self):
        outputs = self._outputs().get()
        prefix = self.inputs.prefix
        outputs['filter_file'] = f'{prefix}_desc-lfofilterEV_bold.nii.gz'
        if self.inputs.glmderivs > 0:
            for i_deriv in range(self.inputs.glmderivs):
                outputs[f'filter_file_deriv{i_deriv + 1}'] = (
                    f'{prefix}_desc-lfofilterEVDeriv{i_deriv + 1}_bold.nii.gz'
                )

        return outputs


class _RetroGLMInputSpec(CommandLineInputSpec):
    in_file = File(
        exists=True,
        argstr='%s',
        position=0,
        mandatory=True,
        desc='File to denoise',
    )
    datafileroot = traits.Str(
        argstr='%s',
        position=1,
        mandatory=True,
        desc=(
            'The root name of the previously run rapidtide dataset '
            '(everything up to but not including the underscore.)'
        ),
    )
    runoptionsfile = File(
        exists=True,
        argstr='--runoptionsfile %s',
        mandatory=False,
    )
    procmaskfile = File(
        exists=True,
        argstr='--procmaskfile %s',
        mandatory=False,
    )
    corrmaskfile = File(
        exists=True,
        argstr='--corrmaskfile %s',
        mandatory=False,
    )
    lagtimesfile = File(
        exists=True,
        argstr='--lagtimesfile %s',
        mandatory=False,
    )
    lagtcgeneratorfile = File(
        exists=True,
        argstr='--lagtcgeneratorfile %s',
        mandatory=False,
    )
    meanfile = File(
        exists=True,
        argstr='--meanfile %s',
        mandatory=False,
    )


class _RetroGLMOutputSpec(TraitedSpec):
    denoised = File(exists=True, desc='Denoised time series')
    denoised_json = File(exists=True, desc='Denoised time series metadata')


class RetroGLM(CommandLine):
    """Run the rapidtide command-line interface."""

    _cmd = 'retroglm'
    input_spec = _RetroGLMInputSpec
    output_spec = _RetroGLMOutputSpec

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
