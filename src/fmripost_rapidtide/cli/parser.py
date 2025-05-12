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
"""Parser."""

import sys

from fmripost_rapidtide import config


def _build_parser(**kwargs):
    """Build parser object.

    ``kwargs`` are passed to ``argparse.ArgumentParser``.
    """

    from argparse import Action, ArgumentDefaultsHelpFormatter, ArgumentParser
    from functools import partial
    from pathlib import Path

    from niworkflows.utils.spaces import OutputReferencesAction
    from packaging.version import Version
    from rapidtide.workflows.rapidtide_parser import (
        DEFAULT_CORRWEIGHTING,
        DEFAULT_DETREND_ORDER,
        DEFAULT_INITREGRESSOR_METHOD,
        DEFAULT_INITREGRESSOR_PCACOMPONENTS,
        DEFAULT_LAGMAX,
        DEFAULT_LAGMAX_THRESH,
        DEFAULT_LAGMIN,
        DEFAULT_LAGMIN_THRESH,
        DEFAULT_MAXPASSES,
        DEFAULT_OUTPUTLEVEL,
        DEFAULT_REFINE_PCACOMPONENTS,
        DEFAULT_REGRESSIONFILTDERIVS,
        DEFAULT_SIGMAMAX,
        DEFAULT_SIGMATHRESH,
        DEFAULT_SPATIALFILT,
    )

    from fmripost_rapidtide.cli.version import check_latest, is_flagged

    class IndicateSpecifiedAction(Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, values)
            setattr(namespace, self.dest + '_nondefault', True)

    class ToDict(Action):
        def __call__(self, parser, namespace, values, option_string=None):
            d = {}
            for spec in values:
                try:
                    name, loc = spec.split('=')
                    loc = Path(loc)
                except ValueError:
                    loc = Path(spec)
                    name = loc.name

                if name in d:
                    raise ValueError(f'Received duplicate derivative name: {name}')

                d[name] = loc
            setattr(namespace, self.dest, d)

    def _path_exists(path, parser):
        """Ensure a given path exists."""
        if path is None or not Path(path).exists():
            raise parser.error(f'Path does not exist: <{path}>.')
        return Path(path).absolute()

    def _is_file(path, parser):
        """Ensure a given path exists and it is a file."""
        path = _path_exists(path, parser)
        if not path.is_file():
            raise parser.error(f'Path should point to a file (or symlink of file): <{path}>.')
        return path

    def _min_one(value, parser):
        """Ensure an argument is not lower than 1."""
        value = int(value)
        if value < 1:
            raise parser.error("Argument can't be less than one.")
        return value

    def _float_or_auto(value, parser):
        """Ensure an argument is a float or 'auto'."""
        if value != 'auto':
            try:
                value = float(value)
            except parser.error:
                raise parser.error(f'Value {value} is not a float or "auto"')
        return value

    def _invert_float_or_auto(value, parser):
        """Ensure an argument is a float or 'auto'."""
        if value != 'auto':
            try:
                value = 1 / float(value)
            except parser.error:
                raise parser.error(f'Value {value} is not a float or "auto"')
        return value

    def _to_gb(value):
        scale = {'G': 1, 'T': 10**3, 'M': 1e-3, 'K': 1e-6, 'B': 1e-9}
        digits = ''.join([c for c in value if c.isdigit()])
        units = value[len(digits) :] or 'M'
        return int(digits) * scale[units[0]]

    def _drop_sub(value):
        return value[4:] if value.startswith('sub-') else value

    def _filter_pybids_none_any(dct):
        import bids

        return {
            k: bids.layout.Query.NONE if v is None else (bids.layout.Query.ANY if v == '*' else v)
            for k, v in dct.items()
        }

    def _bids_filter(value, parser):
        from json import JSONDecodeError, loads

        if value:
            if Path(value).exists():
                try:
                    return loads(Path(value).read_text(), object_hook=_filter_pybids_none_any)
                except JSONDecodeError as e:
                    raise parser.error(f'JSON syntax error in: <{value}>.') from e
            else:
                raise parser.error(f'Path does not exist: <{value}>.')

    verstr = f'fMRIPost-rapidtide v{config.environment.version}'
    currentv = Version(config.environment.version)
    is_release = not any((currentv.is_devrelease, currentv.is_prerelease, currentv.is_postrelease))

    parser = ArgumentParser(
        description=(
            'fMRIPost-rapidtide: fMRI POSTprocessing Rapidtide workflow '
            f'v{config.environment.version}'
        ),
        formatter_class=ArgumentDefaultsHelpFormatter,
        **kwargs,
    )
    PathExists = partial(_path_exists, parser=parser)
    IsFile = partial(_is_file, parser=parser)
    PositiveInt = partial(_min_one, parser=parser)
    BIDSFilter = partial(_bids_filter, parser=parser)

    # Arguments as specified by BIDS-Apps
    # required, positional arguments
    # IMPORTANT: they must go directly with the parser object
    parser.add_argument(
        'bids_dir',
        action='store',
        type=PathExists,
        help=(
            'The root folder of a BIDS-valid raw dataset '
            '(sub-XXXXX folders should be found at the top level in this folder).'
        ),
    )
    parser.add_argument(
        'output_dir',
        action='store',
        type=Path,
        help='The output path for the outcomes of preprocessing and visual reports',
    )
    parser.add_argument(
        'analysis_level',
        choices=['participant'],
        help=(
            "Processing stage to be run, only 'participant' in the case of "
            'fMRIPost-rapidtide (see BIDS-Apps specification).'
        ),
    )

    # Rapidtide options
    # Preprocessing options
    preproc = parser.add_argument_group('Preprocessing options')
    preproc.add_argument(
        '--autosync',
        dest='autosync',
        action='store_true',
        help=(
            'Estimate and apply the initial offsettime of an external '
            'regressor using the global crosscorrelation. '
            'Overrides offsettime if present.'
        ),
        default=False,
    )

    # Add filter options
    filt_opts = parser.add_argument_group('Filtering options')
    filt_opts.add_argument(
        '--filterband',
        dest='filterband',
        action='store',
        type=str,
        choices=[
            'None',
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
        ],
        help='Filter data and regressors to specific band. Use "None" to disable filtering.',
        default='lfo',
    )
    filt_opts.add_argument(
        '--filterfreqs',
        dest='filterfreqs',
        action='store',
        nargs=2,
        type=float,
        metavar=('LOWERPASS', 'UPPERPASS'),
        help=(
            'Filter data and regressors to retain LOWERPASS to UPPERPASS. '
            'If --filterstopfreqs is not also specified, '
            'LOWERSTOP and UPPERSTOP will be calculated automatically.'
        ),
        default=None,
    )
    filt_opts.add_argument(
        '--filterstopfreqs',
        dest='filterstopfreqs',
        action='store',
        nargs=2,
        type=float,
        metavar=('LOWERSTOP', 'UPPERSTOP'),
        help=(
            'Filter data and regressors to with stop frequencies LOWERSTOP and UPPERSTOP. '
            'LOWERSTOP must be <= LOWERPASS, UPPERSTOP must be >= UPPERPASS. '
            'Using this argument requires the use of --filterfreqs.'
        ),
        default=None,
    )

    # Add permutation options
    sigcalc_opts = parser.add_argument_group('Significance calculation options')
    sigcalc_opts.add_argument(
        '--numnull',
        dest='numnull',
        action='store',
        type=int,
        metavar='NREPS',
        help=(
            'Estimate significance threshold by running NREPS null correlations. '
            'Set to 0 to disable.'
        ),
        default=10000,
    )

    preproc.add_argument(
        '--detrendorder',
        dest='detrendorder',
        action='store',
        type=int,
        metavar='ORDER',
        help='Set order of trend removal (0 to disable).',
        default=DEFAULT_DETREND_ORDER,
    )
    preproc.add_argument(
        '--spatialfilt',
        dest='spatialfilt',
        action='store',
        type=float,
        metavar='GAUSSSIGMA',
        help=(
            'Spatially filter fMRI data prior to analysis '
            'using GAUSSSIGMA in mm.  Set GAUSSSIGMA negative '
            'to have rapidtide set it to half the mean voxel '
            'dimension (a rule of thumb for a good value).'
        ),
        default=DEFAULT_SPATIALFILT,
    )
    preproc.add_argument(
        '--confoundfile',
        dest='confoundfile',
        metavar='CONFFILE',
        help=(
            'Read additional (non-motion) confound regressors out of CONFFILE file '
            '(which can be any type of multicolumn text file '
            'rapidtide reads as long as data is sampled at TR with timepoints rows). '
            'Optionally do power expansion and/or calculate derivatives prior to regression.'
        ),
        default=None,
    )
    preproc.add_argument(
        '--confoundpowers',
        dest='confoundpowers',
        metavar='N',
        type=int,
        help='Include powers of each confound regressor up to order N.',
        default=1,
    )
    preproc.add_argument(
        '--noconfoundderiv',
        dest='noconfoundderiv',
        action='store_false',
        help='Toggle whether derivatives will be used in confound regression.',
        default=True,
    )
    preproc.add_argument(
        '--globalsignalmethod',
        dest='globalsignalmethod',
        action='store',
        type=str,
        choices=['sum', 'meanscale', 'pca', 'random'],
        help=(
            'The method for constructing the initial global signal regressor - '
            'straight summation, '
            'mean scaling each voxel prior to summation, '
            'MLE PCA of the voxels in the global signal mask, '
            'or initializing using random noise.'
        ),
        default=DEFAULT_INITREGRESSOR_METHOD,
    )
    preproc.add_argument(
        '--globalpcacomponents',
        dest='globalpcacomponents',
        action='store',
        type=float,
        metavar='VALUE',
        help=(
            'Number of PCA components used for estimating the global signal. '
            'If VALUE >= 1, will retain this many components. '
            'If 0.0 < VALUE < 1.0, enough components will be retained to explain the fraction '
            'VALUE of the total variance. '
            'If VALUE is negative, the number of components will be to retain will be selected '
            'automatically using the MLE method.'
        ),
        default=DEFAULT_INITREGRESSOR_PCACOMPONENTS,
    )
    preproc.add_argument(
        '--numskip',
        dest='numskip',
        action='store',
        type=int,
        metavar='NUMPOINTS',
        help=(
            'When calculating the moving regressor, '
            'set this number of points to zero at the beginning of the voxel timecourses. '
            'This prevents initial points which may not be in equilibrium from contaminating the '
            'calculated sLFO signal. '
            'This may improve similarity fitting and GLM noise removal.'
        ),
        default=0,
    )
    preproc.add_argument(
        '--timerange',
        dest='timerange',
        action='store',
        nargs=2,
        type=int,
        metavar=('START', 'END'),
        help=(
            'Limit analysis to data between timepoints START and END in the fmri file. '
            'If END is set to -1, analysis will go to the last timepoint. '
            'Negative values of START will be set to 0. Default is to use all timepoints.'
        ),
        default=(-1, -1),
    )

    # Correlation options
    corr = parser.add_argument_group('Correlation options')
    corr.add_argument(
        '--corrweighting',
        dest='corrweighting',
        action='store',
        type=str,
        choices=['None', 'phat', 'liang', 'eckart', 'regressor'],
        help=(
            'Method to use for cross-correlation weighting. '
            "'None' performs an unweighted correlation. "
            "'phat' weights the correlation by the magnitude of the product of the "
            "timecourse's FFTs. "
            "'liang' weights the correlation by the sum of the magnitudes of the "
            "timecourse's FFTs. "
            "'eckart' weights the correlation by the product of the magnitudes of the "
            "timecourse's FFTs. "
            "'regressor' weights the correlation by the magnitude of the sLFO regressor FFT."
        ),
        default=DEFAULT_CORRWEIGHTING,
    )
    corr.add_argument(
        '--simcalcrange',
        dest='simcalcrange',
        action='store',
        nargs=2,
        type=int,
        metavar=('START', 'END'),
        help=(
            'Limit correlation calculation to data between timepoints START and END in the fmri '
            'file. '
            'If END is set to -1, analysis will go to the last timepoint. '
            'Negative values of START will be set to 0. '
            'Default is to use all timepoints. '
            'NOTE: these offsets are relative to the start of the dataset AFTER any trimming '
            "done with '--timerange'."
        ),
        default=(-1, -1),
    )

    # Correlation fitting options
    corr_fit = parser.add_argument_group('Correlation fitting options')
    fixdelay = corr_fit.add_mutually_exclusive_group()
    fixdelay.add_argument(
        '--fixdelay',
        dest='fixdelay',
        action='store',
        type=float,
        metavar='DELAYTIME',
        help="Don't fit the delay time - set it to DELAYTIME seconds for all voxels.",
        default=None,
    )
    fixdelay.add_argument(
        '--searchrange',
        dest='searchrange',
        action=IndicateSpecifiedAction,
        nargs=2,
        type=float,
        metavar=('LAGMIN', 'LAGMAX'),
        help='Limit fit to a range of lags from LAGMIN to LAGMAX.',
        default=(DEFAULT_LAGMIN, DEFAULT_LAGMAX),
    )
    corr_fit.add_argument(
        '--sigmalimit',
        dest='sigmalimit',
        action='store',
        type=float,
        metavar='SIGMALIMIT',
        help='Reject lag fits with linewidth wider than SIGMALIMIT Hz.',
        default=DEFAULT_SIGMAMAX,
    )
    corr_fit.add_argument(
        '--bipolar',
        dest='bipolar',
        action='store_true',
        help='Bipolar mode - match peak correlation ignoring sign.',
        default=False,
    )

    # Regressor refinement options
    reg_ref = parser.add_argument_group('Regressor refinement options')
    reg_ref.add_argument(
        '--lagminthresh',
        dest='lagminthresh',
        action='store',
        metavar='MIN',
        type=float,
        help='For refinement, exclude voxels with delays less than MIN.',
        default=DEFAULT_LAGMIN_THRESH,
    )
    reg_ref.add_argument(
        '--lagmaxthresh',
        dest='lagmaxthresh',
        action='store',
        metavar='MAX',
        type=float,
        help='For refinement, exclude voxels with delays greater than MAX.',
        default=DEFAULT_LAGMAX_THRESH,
    )
    reg_ref.add_argument(
        '--ampthresh',
        dest='ampthresh',
        action='store',
        metavar='AMP',
        type=float,
        help=(
            'For refinement, exclude voxels with correlation coefficients less than AMP. '
            'NOTE: ampthresh will automatically be set to the p<0.05 '
            'significance level determined by the --numnull option if NREPS '
            'is set greater than 0 and this is not manually specified.'
        ),
        default=-1.0,
    )
    reg_ref.add_argument(
        '--sigmathresh',
        dest='sigmathresh',
        action='store',
        metavar='SIGMA',
        type=float,
        help='For refinement, exclude voxels with widths greater than SIGMA seconds.',
        default=DEFAULT_SIGMATHRESH,
    )
    reg_ref.add_argument(
        '--pcacomponents',
        dest='pcacomponents',
        action='store',
        type=float,
        metavar='VALUE',
        help=(
            'Number of PCA components used for refinement. '
            'If VALUE >= 1, will retain this many components. '
            'If 0.0 < VALUE < 1.0, enough components will be retained to explain the fraction '
            'VALUE of the total variance. '
            'If VALUE is negative, the number of components will be to retain will be selected '
            'automatically using the MLE method.'
        ),
        default=DEFAULT_REFINE_PCACOMPONENTS,
    )
    reg_ref.add_argument(
        '--convergencethresh',
        dest='convergencethresh',
        action='store',
        type=float,
        metavar='THRESH',
        help=(
            'Continue refinement until the MSE between regressors becomes <= THRESH.  '
            'By default, this is not set, '
            'so refinement will run for the specified number of passes. '
        ),
        default=None,
    )
    reg_ref.add_argument(
        '--maxpasses',
        dest='maxpasses',
        action='store',
        type=int,
        metavar='MAXPASSES',
        help=(
            'Terminate refinement after MAXPASSES passes, whether or not convergence has occurred.'
        ),
        default=DEFAULT_MAXPASSES,
    )

    # GLM noise removal options
    glm = parser.add_argument_group('GLM noise removal options')
    glm.add_argument(
        '--glmsourcefile',
        dest='glmsourcefile',
        action='store',
        type=IsFile,
        metavar='FILE',
        help=(
            'Regress delayed regressors out of FILE instead of the initial fmri file used to '
            'estimate delays.'
        ),
        default=None,
    )
    glm.add_argument(
        '--regressderivs',
        dest='regressderivs',
        action='store',
        type=int,
        metavar='NDERIVS',
        help='When doing final GLM, include derivatives up to NDERIVS order.',
        default=DEFAULT_REGRESSIONFILTDERIVS,
    )

    # Output options
    output = parser.add_argument_group('Output options')
    output.add_argument(
        '--outputlevel',
        dest='outputlevel',
        action='store',
        type=str,
        choices=['min', 'less', 'normal', 'more', 'max'],
        help=(
            'The level of file output produced.  '
            "'min' produces only absolutely essential files, "
            "'less' adds in the GLM filtered data (rather than just filter efficacy metrics), "
            "'normal' saves what you would typically want around for interactive data "
            'exploration, '
            "'more' adds files that are sometimes useful, and 'max' outputs anything you "
            'might possibly want. '
            "Selecting 'max' will produce ~3x your input datafile size as output."
        ),
        default=DEFAULT_OUTPUTLEVEL,
    )

    # Experimental options (not fully tested, may not work)
    experimental = parser.add_argument_group(
        'Experimental options (not fully tested, or not tested at all, may not work).  Beware!'
    )
    experimental.add_argument(
        '--territorymap',
        dest='territorymap',
        metavar='MAP[:VALSPEC]',
        help=(
            'This specifies a territory map. '
            'Each territory is a set of voxels with the same integral value. '
            'If VALSPEC is given, only territories in the mask with integral values listed in '
            'VALSPEC are used, otherwise all nonzero voxels are used. '
            'If this option is set, certain output measures will be summarized over each '
            'territory in the map, in addition to over the whole brain. '
            'Some interesting territory maps might be: '
            'a gray/white/csf segmentation image, an arterial territory map, '
            'lesion area vs. healthy tissue segmentation, etc. '
            "NB: at the moment this is just a placeholder - it doesn't do anything."
        ),
        default=None,
    )
    experimental.add_argument(
        '--autorespdelete',
        dest='autorespdelete',
        action='store_true',
        help='Attempt to detect and remove respiratory signal that strays into the LFO band.',
        default=False,
    )
    # End of rapidtide options

    g_bids = parser.add_argument_group('Options for filtering BIDS queries')
    g_bids.add_argument(
        '--skip_bids_validation',
        '--skip-bids-validation',
        action='store_true',
        default=False,
        help='Assume the input dataset is BIDS compliant and skip the validation',
    )
    g_bids.add_argument(
        '--participant-label',
        '--participant_label',
        action='store',
        nargs='+',
        type=_drop_sub,
        help=(
            'A space delimited list of participant identifiers or a single '
            'identifier (the sub- prefix can be removed)'
        ),
    )
    g_bids.add_argument(
        '-t',
        '--task-id',
        action='store',
        help='Select a specific task to be processed',
    )
    g_bids.add_argument(
        '--bids-filter-file',
        dest='bids_filters',
        action='store',
        type=BIDSFilter,
        metavar='FILE',
        help=(
            'A JSON file describing custom BIDS input filters using PyBIDS. '
            'For further details, please check out '
            'https://fmriprep.readthedocs.io/en/'
            f'{currentv.base_version if is_release else "latest"}/faq.html#'
            'how-do-I-select-only-certain-files-to-be-input-to-fMRIPrep'
        ),
    )
    g_bids.add_argument(
        '-d',
        '--derivatives',
        action=ToDict,
        metavar='PACKAGE=PATH',
        nargs='+',
        help=(
            'Search PATH(s) for pre-computed derivatives. '
            'These may be provided as named folders '
            '(e.g., `--derivatives smriprep=/path/to/smriprep`).'
        ),
    )
    g_bids.add_argument(
        '--bids-database-dir',
        metavar='PATH',
        type=Path,
        help=(
            'Path to a PyBIDS database folder, for faster indexing '
            '(especially useful for large datasets). '
            'Will be created if not present.'
        ),
    )

    g_perfm = parser.add_argument_group('Options to handle performance')
    g_perfm.add_argument(
        '--nprocs',
        '--nthreads',
        '--n_cpus',
        '--n-cpus',
        dest='nprocs',
        action='store',
        type=PositiveInt,
        help='Maximum number of threads across all processes',
    )
    g_perfm.add_argument(
        '--omp-nthreads',
        action='store',
        type=PositiveInt,
        help='Maximum number of threads per-process',
    )
    g_perfm.add_argument(
        '--mem',
        '--mem_mb',
        '--mem-mb',
        dest='memory_gb',
        action='store',
        type=_to_gb,
        metavar='MEMORY_MB',
        help='Upper bound memory limit for fMRIPost-rapidtide processes',
    )
    g_perfm.add_argument(
        '--low-mem',
        action='store_true',
        help='Attempt to reduce memory usage (will increase disk usage in working directory)',
    )
    g_perfm.add_argument(
        '--use-plugin',
        '--nipype-plugin-file',
        action='store',
        metavar='FILE',
        type=IsFile,
        help='Nipype plugin configuration file',
    )
    g_perfm.add_argument(
        '--sloppy',
        action='store_true',
        default=False,
        help='Use low-quality tools for speed - TESTING ONLY',
    )

    g_subset = parser.add_argument_group('Options for performing only a subset of the workflow')
    g_subset.add_argument(
        '--boilerplate-only',
        '--boilerplate_only',
        action='store_true',
        default=False,
        help='Generate boilerplate only',
    )
    g_subset.add_argument(
        '--reports-only',
        action='store_true',
        default=False,
        help=(
            "Only generate reports, don't run workflows. "
            'This will only rerun report aggregation, not reportlet generation for specific '
            'nodes.'
        ),
    )

    g_conf = parser.add_argument_group('Workflow configuration')
    g_conf.add_argument(
        '--ignore',
        required=False,
        action='store',
        nargs='+',
        default=[],
        choices=['fieldmaps', 'slicetiming', 'jacobian'],
        help=(
            'Ignore selected aspects of the input dataset to disable corresponding '
            'parts of the resampling workflow (a space delimited list)'
        ),
    )
    g_conf.add_argument(
        '--output-spaces',
        nargs='*',
        action=OutputReferencesAction,
        help="""\
    Standard and non-standard spaces to resample denoised functional images to. \
    Standard spaces may be specified by the form \
    ``<SPACE>[:cohort-<label>][:res-<resolution>][...]``, where ``<SPACE>`` is \
    a keyword designating a spatial reference, and may be followed by optional, \
    colon-separated parameters. \
    Non-standard spaces imply specific orientations and sampling grids. \
    For further details, please check out \
    https://fmriprep.readthedocs.io/en/%s/spaces.html"""
       % (currentv.base_version if is_release else 'latest'),
    )
    g_conf.add_argument(
        '--dummy-scans',
        required=False,
        action='store',
        default=None,
        type=int,
        help='Number of nonsteady-state volumes. Overrides automatic detection.',
    )
    g_conf.add_argument(
        '--random-seed',
        dest='_random_seed',
        action='store',
        type=int,
        default=None,
        help='Initialize the random seed for the workflow',
    )

    g_outputs = parser.add_argument_group('Options for modulating outputs')
    g_outputs.add_argument(
        '--md-only-boilerplate',
        action='store_true',
        default=False,
        help='Skip generation of HTML and LaTeX formatted citation with pandoc',
    )
    g_outputs.add_argument(
        '--aggregate-session-reports',
        dest='aggr_ses_reports',
        action='store',
        type=PositiveInt,
        default=4,
        help=(
            "Maximum number of sessions aggregated in one subject's visual report. "
            'If exceeded, visual reports are split by session.'
        ),
    )

    g_carbon = parser.add_argument_group('Options for carbon usage tracking')
    g_carbon.add_argument(
        '--track-carbon',
        action='store_true',
        help='Tracks power draws using CodeCarbon package',
    )
    g_carbon.add_argument(
        '--country-code',
        action='store',
        default='CAN',
        type=str,
        help='Country ISO code used by carbon trackers',
    )

    g_other = parser.add_argument_group('Other options')
    g_other.add_argument('--version', action='version', version=verstr)
    g_other.add_argument(
        '-v',
        '--verbose',
        dest='verbose_count',
        action='count',
        default=0,
        help='Increases log verbosity for each occurrence, debug level is -vvv',
    )
    g_other.add_argument(
        '-w',
        '--work-dir',
        action='store',
        type=Path,
        default=Path('work').absolute(),
        help='Path where intermediate results should be stored',
    )
    g_other.add_argument(
        '--clean-workdir',
        action='store_true',
        default=False,
        help='Clears working directory of contents. Use of this flag is not '
        'recommended when running concurrent processes of fMRIPost-rapidtide.',
    )
    g_other.add_argument(
        '--resource-monitor',
        action='store_true',
        default=False,
        help="Enable Nipype's resource monitoring to keep track of memory and CPU usage",
    )
    g_other.add_argument(
        '--config-file',
        action='store',
        metavar='FILE',
        help='Use pre-generated configuration file. Values in file will be overridden '
        'by command-line arguments.',
    )
    g_other.add_argument(
        '--write-graph',
        action='store_true',
        default=False,
        help='Write workflow graph.',
    )
    g_other.add_argument(
        '--stop-on-first-crash',
        action='store_true',
        default=False,
        help='Force stopping on first crash, even if a work directory was specified.',
    )
    g_other.add_argument(
        '--notrack',
        action='store_true',
        default=False,
        help='Opt-out of sending tracking information of this run to '
        'the FMRIPREP developers. This information helps to '
        'improve FMRIPREP and provides an indicator of real '
        'world usage crucial for obtaining funding.',
    )
    g_other.add_argument(
        '--debug',
        action='store',
        nargs='+',
        choices=config.DEBUG_MODES + ('all',),
        help="Debug mode(s) to enable. 'all' is alias for all available modes.",
    )

    latest = check_latest()
    if latest is not None and currentv < latest:
        print(
            f"""\
You are using fMRIPost-rapidtide-{currentv},
and a newer version of fMRIPost-rapidtide is available: {latest}.
Please check out our documentation about how and when to upgrade:
https://fmriprep.readthedocs.io/en/latest/faq.html#upgrading""",
            file=sys.stderr,
        )

    _blist = is_flagged()
    if _blist[0]:
        _reason = _blist[1] or 'unknown'
        print(
            f"""\
WARNING: Version {config.environment.version} of fMRIPost-rapidtide (current) has been FLAGGED
(reason: {_reason}).
That means some severe flaw was found in it and we strongly
discourage its usage.""",
            file=sys.stderr,
        )

    return parser


def parse_args(args=None, namespace=None):
    """Parse args and run further checks on the command line."""

    import logging

    parser = _build_parser()
    opts = parser.parse_args(args, namespace)

    if opts.config_file:
        skip = {} if opts.reports_only else {'execution': ('run_uuid',)}
        config.load(opts.config_file, skip=skip, init=False)
        config.loggers.cli.info(f'Loaded previous configuration file {opts.config_file}')

    config.execution.log_level = int(max(25 - 5 * opts.verbose_count, logging.DEBUG))
    config.from_dict(vars(opts), init=['nipype'])

    if not config.execution.notrack:
        import importlib.util

        if importlib.util.find_spec('sentry_sdk') is None:
            config.execution.notrack = True
            config.loggers.cli.warning('Telemetry disabled because sentry_sdk is not installed.')
        else:
            config.loggers.cli.info(
                'Telemetry system to collect crashes and errors is enabled '
                '- thanks for your feedback!. Use option ``--notrack`` to opt out.'
            )

    # Retrieve logging level
    build_log = config.loggers.cli

    # Load base plugin_settings from file if --use-plugin
    if opts.use_plugin is not None:
        import yaml

        with open(opts.use_plugin) as f:
            plugin_settings = yaml.safe_load(f)
        _plugin = plugin_settings.get('plugin')
        if _plugin:
            config.nipype.plugin = _plugin
            config.nipype.plugin_args = plugin_settings.get('plugin_args', {})
            config.nipype.nprocs = opts.nprocs or config.nipype.plugin_args.get(
                'n_procs', config.nipype.nprocs
            )

    # Resource management options
    # Note that we're making strong assumptions about valid plugin args
    # This may need to be revisited if people try to use batch plugins
    if 1 < config.nipype.nprocs < config.nipype.omp_nthreads:
        build_log.warning(
            f'Per-process threads (--omp-nthreads={config.nipype.omp_nthreads}) exceed '
            f'total threads (--nthreads/--n_cpus={config.nipype.nprocs})'
        )

    bids_dir = config.execution.bids_dir
    output_dir = config.execution.output_dir
    work_dir = config.execution.work_dir
    version = config.environment.version

    # Wipe out existing work_dir
    if opts.clean_workdir and work_dir.exists():
        from niworkflows.utils.misc import clean_directory

        build_log.info(f'Clearing previous fMRIPost-rapidtide working directory: {work_dir}')
        if not clean_directory(work_dir):
            build_log.warning(f'Could not clear all contents of working directory: {work_dir}')

    # Update the config with an empty dict to trigger initialization of all config
    # sections (we used `init=False` above).
    # This must be done after cleaning the work directory, or we could delete an
    # open SQLite database
    config.from_dict({})

    # Ensure input and output folders are not the same
    if output_dir == bids_dir:
        parser.error(
            'The selected output folder is the same as the input BIDS folder. '
            'Please modify the output path (suggestion: '
            f'{bids_dir / "derivatives" / "fmripost_rapidtide-" + version.split("+")[0]}.'
        )

    if bids_dir in work_dir.parents:
        parser.error(
            'The selected working directory is a subdirectory of the input BIDS folder. '
            'Please modify the output path.'
        )

    # Validate inputs
    if not opts.skip_bids_validation:
        from fmripost_rapidtide.utils.bids import validate_input_dir

        build_log.info(
            'Making sure the input data is BIDS compliant (warnings can be ignored in most cases).'
        )
        validate_input_dir(config.environment.exec_env, opts.bids_dir, opts.participant_label)

    # Setup directories
    config.execution.log_dir = config.execution.output_dir / 'logs'
    # Check and create output and working directories
    config.execution.log_dir.mkdir(exist_ok=True, parents=True)
    work_dir.mkdir(exist_ok=True, parents=True)

    # Force initialization of the BIDSLayout
    config.execution.init()
    all_subjects = config.execution.layout.get_subjects()
    if config.execution.participant_label is None:
        config.execution.participant_label = all_subjects

    participant_label = set(config.execution.participant_label)
    missing_subjects = participant_label - set(all_subjects)
    if missing_subjects:
        parser.error(
            'One or more participant labels were not found in the BIDS directory: '
            f'{", ".join(missing_subjects)}.'
        )

    config.execution.participant_label = sorted(participant_label)
