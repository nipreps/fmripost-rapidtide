"""Interfaces to run rapidtide."""

import os

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
        1,
        usedefault=True,
        argstr='--nprocs %d',
        mandatory=False,
    )


class _RapidtideOutputSpec(TraitedSpec):
    prefix = traits.Str(desc='Directory containing the results, with prefix.')
    rapidtide_dir = traits.Directory(
        desc='Directory containing the results.',
        exists=True,
    )
    maxtimemap = File(
        exists=True,
        desc='3D map of optimal delay times (usually called XXX_desc-maxtime_map.nii.gz)',
    )
    maxtimemap_json = File(
        exists=True,
        desc='3D map of optimal delay times sidecar (usually called XXX_desc-maxtime_map.json)',
    )
    lagtcgenerator = File(
        exists=True,
        desc=(
            'Time series of refined regressor (usually called '
            'XXX_desc-lagtcgenerator_timeseries.tsv)'
        ),
    )
    lagtcgenerator_json = File(
        exists=True,
        desc=(
            'Time series of refined regressor sidecar file (usually called '
            'XXX_desc-lagtcgenerator_timeseries.json)'
        ),
    )
    strengthmap = File(
        exists=True,
        desc='Strength map (usually called XXX_desc-maxcorr_map.nii.gz)',
    )
    strengthmap_json = File(
        exists=True,
        desc='Strength map sidecar file (usually called XXX_desc-maxcorr_map.json)',
    )
    slfoamplitude = File(
        exists=True,
        desc=(
            'Time series of sLFO amplitude (usually called XXX_desc-sLFOamplitude_timeseries.tsv)'
        ),
    )
    slfoamplitude_json = File(
        exists=True,
        desc=(
            'Time series of sLFO amplitude sidecar file (usually called '
            'XXX_desc-sLFOamplitude_timeseries.json)'
        ),
    )
    delayrankordermap = File(
        exists=True,
        desc='Delay rank order map (usually called XXX_desc-timepercentile_map.nii.gz)',
    )
    delayrankordermap_json = File(
        exists=True,
        desc='Delay rank order map sidecar file (usually called XXX_desc-timepercentile_map.json)',
    )
    correlationwidthmap = File(
        exists=True,
        desc='Correlation width map (usually called XXX_desc-maxwidth_map.nii.gz)',
    )
    correlationwidthmap_json = File(
        exists=True,
        desc='Correlation width map sidecar file (usually called XXX_desc-maxwidth_map.json)',
    )
    maskfile = File(exists=True, desc='Mask file (usually called XXX_desc-corrfit_mask.nii.gz)')
    runoptions = File(exists=True, desc='Run options json (XXX_desc-runoptions_info.json)')


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
        outputs['maxtimemap'] = f'{prefix}_desc-maxtime_map.nii.gz'
        outputs['maxtimemap_json'] = f'{prefix}_desc-maxtime_map.json'
        outputs['lagtcgenerator'] = f'{prefix}_desc-lagtcgenerator_timeseries.tsv.gz'
        outputs['lagtcgenerator_json'] = f'{prefix}_desc-lagtcgenerator_timeseries.json'
        outputs['strengthmap'] = f'{prefix}_desc-maxcorr_map.nii.gz'
        outputs['strengthmap_json'] = f'{prefix}_desc-maxcorr_map.json'
        outputs['slfoamplitude'] = f'{prefix}_desc-sLFOamplitude_timeseries.tsv'
        outputs['slfoamplitude_json'] = f'{prefix}_desc-sLFOamplitude_timeseries.json'
        outputs['delayrankordermap'] = f'{prefix}_desc-timepercentile_map.nii.gz'
        outputs['delayrankordermap_json'] = f'{prefix}_desc-timepercentile_map.json'
        outputs['correlationwidthmap'] = f'{prefix}_desc-maxwidth_map.nii.gz'
        outputs['correlationwidthmap_json'] = f'{prefix}_desc-maxwidth_map.json'
        outputs['maskfile'] = f'{prefix}_desc-corrfit_mask.nii.gz'
        outputs['runoptions'] = f'{prefix}_desc-runoptions_info.json'
        outputs['rapidtide_dir'] = os.getcwd()

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
        0,
        argstr='--glmderivs %d',
        mandatory=False,
        desc='When doing final GLM, include derivatives up to NDERIVS order. Default is 0.',
        usedefault=True,
    )
    nprocs = traits.Int(
        1,
        usedefault=True,
        argstr='--nprocs %d',
        mandatory=False,
        desc=(
            'Use NPROCS worker processes for multiprocessing. '
            'Setting NPROCS to less than 1 sets the number of worker processes to n_cpus.'
        ),
    )
    numskip = traits.Int(
        0,
        argstr='--numskip %d',
        usedefault=True,
        mandatory=False,
        desc='Skip NUMSKIP points at the beginning of the fmri file.',
    )
    noprogressbar = traits.Bool(
        True,
        argstr='--noprogressbar',
        mandatory=False,
        usedefault=True,
        desc='Will disable showing progress bars (helpful if stdout is going to a file).',
    )
    debug = traits.Bool(
        False,
        argstr='--debug',
        mandatory=False,
        usedefault=True,
        desc='Output lots of helpful information.',
    )


class _RetroLagTCSOutputSpec(DynamicTraitedSpec):
    filter_file = File(exists=True, desc='Filter file')
    filter_json = File(exists=True, desc='Filter file json')


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
        outputs['filter_json'] = f'{prefix}_desc-lfofilterEV_bold.json'
        if self.inputs.glmderivs > 0:
            for i_deriv in range(self.inputs.glmderivs):
                outputs[f'filter_file_deriv{i_deriv + 1}'] = (
                    f'{prefix}_desc-lfofilterEVDeriv{i_deriv + 1}_bold.nii.gz'
                )

        return outputs


class _RetroRegressInputSpec(CommandLineInputSpec):
    in_file = File(
        exists=True,
        argstr='%s',
        position=0,
        mandatory=True,
        desc='File to denoise',
    )
    datafileroot = traits.Directory(
        argstr='%s',
        position=1,
        mandatory=True,
        desc=(
            'The root name of the previously run rapidtide dataset '
            '(everything up to but not including the underscore.)'
        ),
    )
    prefix = traits.Str(
        argstr='--alternateoutput %s',
        mandatory=False,
        genfile=True,
        desc='Output name',
    )
    glmderivs = traits.Int(
        0,
        argstr='--glmderivs %d',
        usedefault=True,
        mandatory=False,
        desc='When doing final GLM, include derivatives up to NDERIVS order.',
    )
    nprocs = traits.Int(
        1,
        usedefault=True,
        argstr='--nprocs %d',
        mandatory=False,
    )
    numskip = traits.Int(
        0,
        usedefault=True,
        argstr='--numskip %d',
        mandatory=False,
    )


class _RetroRegressOutputSpec(TraitedSpec):
    denoised = File(exists=True, desc='Denoised time series')
    denoised_json = File(exists=True, desc='Denoised time series metadata')
    variancechange = File(exists=True, desc='Variance change map')
    variancechange_json = File(exists=True, desc='Variance change map metadata')


class RetroRegress(CommandLine):
    """Run the retroregress CLI to denoise BOLD with existing rapidtide outputs."""

    _cmd = 'retroregress --noprogressbar'
    input_spec = _RetroRegressInputSpec
    output_spec = _RetroRegressOutputSpec

    def _gen_filename(self, name):
        if name == 'prefix':
            return os.path.join(os.getcwd(), 'retroregress')

        return None

    def _list_outputs(self):
        outputs = self._outputs().get()
        prefix_dir = self.inputs.prefix
        datafileroot = self.inputs.datafileroot
        file_prefix = os.path.basename(datafileroot)
        outputs['denoised'] = os.path.join(
            prefix_dir, f'{file_prefix}_desc-lfofilterCleaned_bold.nii.gz'
        )
        outputs['denoised_json'] = os.path.join(
            prefix_dir, f'{file_prefix}_desc-lfofilterCleaned_bold.json'
        )
        outputs['variancechange'] = os.path.join(
            prefix_dir, f'{file_prefix}_desc-lfofilterInbandVarianceChange_map.nii.gz'
        )
        outputs['variancechange_json'] = os.path.join(
            prefix_dir, f'{file_prefix}_desc-lfofilterInbandVarianceChange_map.json'
        )
        return outputs
