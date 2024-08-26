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

    from packaging.version import Version

    from fmripost_rapidtide.cli.version import check_latest, is_flagged

    # from niworkflows.utils.spaces import OutputReferencesAction

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
            f'fMRIPost-rapidtide: fMRI POSTprocessing Rapidtide workflow v{config.environment.version}'
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
            "A JSON file describing custom BIDS input filters using PyBIDS. "
            "For further details, please check out "
            "https://fmriprep.readthedocs.io/en/"
            f"{currentv.base_version if is_release else 'latest'}/faq.html#"
            "how-do-I-select-only-certain-files-to-be-input-to-fMRIPrep"
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
    # Disable output spaces until warping works
    # g_conf.add_argument(
    #     '--output-spaces',
    #     nargs='*',
    #     action=OutputReferencesAction,
    #     help="""\
    # Standard and non-standard spaces to resample denoised functional images to. \
    # Standard spaces may be specified by the form \
    # ``<SPACE>[:cohort-<label>][:res-<resolution>][...]``, where ``<SPACE>`` is \
    # a keyword designating a spatial reference, and may be followed by optional, \
    # colon-separated parameters. \
    # Non-standard spaces imply specific orientations and sampling grids. \
    # For further details, please check out \
    # https://fmriprep.readthedocs.io/en/%s/spaces.html"""
    #    % (currentv.base_version if is_release else 'latest'),
    # )
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

    g_rapidtide = parser.add_argument_group('Options for running rapidtide')
    # Analysis types
    analysis_type = g_rapidtide.add_argument_group(
        title='Analysis type',
        description=(
            'Single arguments that set several parameter values, '
            'tailored to particular analysis types. '
            'Any parameter set by an analysis type can be overridden '
            'by setting that parameter explicitly. '
            'Analysis types are mutually exclusive with one another.'
        ),
    ).add_mutually_exclusive_group()
    analysis_type.add_argument(
        '--denoising',
        dest='denoising',
        action='store_true',
        help=(
            'Preset for hemodynamic denoising - this is a macro that '
            f'sets searchrange=({DEFAULT_DENOISING_LAGMIN}, {DEFAULT_DENOISING_LAGMAX}), '
            f'passes={DEFAULT_DENOISING_PASSES}, despeckle_passes={DEFAULT_DENOISING_DESPECKLE_PASSES}, '
            f'refineoffset=True, peakfittype={DEFAULT_DENOISING_PEAKFITTYPE}, '
            f'gausssigma={DEFAULT_DENOISING_SPATIALFILT}, nofitfilt=True, doglmfilt=True. '
            'Any of these options can be overridden with the appropriate '
            'additional arguments.'
        ),
        default=False,
    )
    analysis_type.add_argument(
        '--delaymapping',
        dest='delaymapping',
        action='store_true',
        help=(
            "Preset for delay mapping analysis - this is a macro that "
            f"sets searchrange=({DEFAULT_DELAYMAPPING_LAGMIN}, {DEFAULT_DELAYMAPPING_LAGMAX}), "
            f"passes={DEFAULT_DELAYMAPPING_PASSES}, "
            f"despeckle_passes={DEFAULT_DELAYMAPPING_DESPECKLE_PASSES}, "
            "refineoffset=True, outputlevel='normal', "
            "doglmfilt=False. "
            "Any of these options can be overridden with the appropriate "
            "additional arguments."
        ),
        default=False,
    )
    analysis_type.add_argument(
        '--CVR',
        dest='docvrmap',
        action='store_true',
        help=(
            'Preset for calibrated CVR mapping.  Given an input regressor that represents some measured '
            'quantity over time (e.g. mmHg CO2 in the EtCO2 trace), rapidtide will calculate and output a map of percent '
            'BOLD change in units of the input regressor.  To do this, this sets: '
            f'passes=1, despeckle_passes={DEFAULT_CVRMAPPING_DESPECKLE_PASSES}, '
            f'searchrange=({DEFAULT_CVRMAPPING_LAGMIN}, {DEFAULT_CVRMAPPING_LAGMAX}), '
            f'filterfreqs=({DEFAULT_CVRMAPPING_FILTER_LOWERPASS}, {DEFAULT_CVRMAPPING_FILTER_UPPERPASS}), '
            'and calculates a voxelwise GLM using the optimally delayed '
            'input regressor and the percent normalized, demeaned BOLD data as inputs. This map is output as '
            '(XXX_desc-CVR_map.nii.gz).  If no input regressor is supplied, this will generate an error.  '
            'These options can be overridden with the appropriate additional arguments.'
        ),
        default=False,
    )
    analysis_type.add_argument(
        '--globalpreselect',
        dest='globalpreselect',
        action='store_true',
        help=(
            "Treat this run as an initial pass to locate good candidate voxels "
            "for global mean regressor generation. "
            "This sets: passes=1, despecklepasses=0, "
            "refinedespeckle=False, outputlevel='normal', doglmfilt=False, "
            "saveintermediatemaps=False."
        ),
        default=False,
    )

    # Macros
    macros = g_rapidtide.add_argument_group(
        title='Macros',
        description=(
            'Single arguments that change default values for many '
            'arguments. '
            'Macros override individually set parameters. '
        ),
    ).add_mutually_exclusive_group()
    macros.add_argument(
        '--venousrefine',
        dest='venousrefine',
        action='store_true',
        help=(
            'This is a macro that sets lagminthresh=2.5, '
            'lagmaxthresh=6.0, ampthresh=0.5, and '
            'refineupperlag to bias refinement towards '
            'voxels in the draining vasculature for an '
            'fMRI scan.'
        ),
        default=False,
    )
    macros.add_argument(
        '--nirs',
        dest='nirs',
        action='store_true',
        help=(
            'This is a NIRS analysis - this is a macro that '
            'sets nothresh, refineprenorm=var, ampthresh=0.7, and '
            'lagminthresh=0.1. '
        ),
        default=False,
    )

    anatomy = g_rapidtide.add_argument_group(
        title='Anatomic information',
        description=(
            "These options allow you to tailor the analysis with some anatomic constraints.  You don't need to supply "
            "any of them, but if you do, rapidtide will try to make intelligent processing decisions based on "
            "these maps.  Any individual masks set with anatomic information will be overridden if you specify "
            "that mask directly."
        ),
    )
    anatomy.add_argument(
        '--brainmask',
        dest='brainmaskincludespec',
        metavar='MASK[:VALSPEC]',
        help=(
            'This specifies the valid brain voxels.  No voxels outside of this mask will be used for global mean '
            'calculation, correlation, refinement, offset calculation, or denoising. '
            'If VALSPEC is given, only voxels in the mask with integral values listed in VALSPEC are used, otherwise '
            'voxels with value > 0.1 are used.  If this option is set, '
            'rapidtide will limit the include mask used to 1) calculate the initial global mean regressor, '
            '2) decide which voxels in which to calculate delays, '
            '3) refine the regressor at the end of each pass, 4) determine the zero time offset value, and 5) process '
            'to remove sLFO signal. '
            'Setting --globalmeaninclude, --refineinclude, --corrmaskinclude or --offsetinclude explicitly will '
            'override this for the given include mask.'
        ),
        default=None,
    )
    anatomy.add_argument(
        '--graymattermask',
        dest='graymatterincludespec',
        metavar='MASK[:VALSPEC]',
        help=(
            'This specifies a gray matter mask registered to the input functional data.  '
            'If VALSPEC is given, only voxels in the mask with integral values listed in VALSPEC are used, otherwise '
            'voxels with value > 0.1 are used.  If this option is set, '
            'rapidtide will use voxels in the gray matter mask to 1) calculate the initial global mean regressor, '
            'and 2) for determining the zero time offset value. '
            'Setting --globalmeaninclude or --offsetinclude explicitly will override this for '
            'the given include mask.'
        ),
        default=None,
    )
    anatomy.add_argument(
        '--whitemattermask',
        dest='whitematterincludespec',
        metavar='MASK[:VALSPEC]',
        help=(
            "This specifies a white matter mask registered to the input functional data.  "
            "If VALSPEC is given, only voxels in the mask with integral values listed in VALSPEC are used, otherwise "
            "voxels with value > 0.1 are used.  "
            "This currently isn't used for anything, but rapidtide will keep track of it and might use if for something "
            "in a later version."
        ),
        default=None,
    )

    # Preprocessing options
    preproc = g_rapidtide.add_argument_group('Preprocessing options')
    realtr = preproc.add_mutually_exclusive_group()
    realtr.add_argument(
        '--datatstep',
        dest='realtr',
        action='store',
        metavar='TSTEP',
        type=lambda x: pf.is_float(parser, x),
        help=(
            'Set the timestep of the data file to TSTEP. '
            'This will override the TR in an '
            'fMRI file. NOTE: if using data from a text '
            'file, for example with NIRS data, using one '
            'of these options is mandatory. '
        ),
        default='auto',
    )
    realtr.add_argument(
        '--datafreq',
        dest='realtr',
        action='store',
        metavar='FREQ',
        type=lambda x: pf.invert_float(parser, x),
        help=(
            'Set the timestep of the data file to 1/FREQ. '
            'This will override the TR in an '
            'fMRI file. NOTE: if using data from a text '
            'file, for example with NIRS data, using one '
            'of these options is mandatory. '
        ),
        default='auto',
    )
    preproc.add_argument(
        '--noantialias',
        dest='antialias',
        action='store_false',
        help='Disable antialiasing filter. ',
        default=True,
    )
    preproc.add_argument(
        '--invert',
        dest='invertregressor',
        action='store_true',
        help=('Invert the sign of the regressor before processing.'),
        default=False,
    )
    preproc.add_argument(
        '--interptype',
        dest='interptype',
        action='store',
        type=str,
        choices=['univariate', 'cubic', 'quadratic'],
        help=(
            'Use specified interpolation type. Options '
            'are "cubic", "quadratic", and "univariate". '
            f'Default is {DEFAULT_INTERPTYPE}. '
        ),
        default=DEFAULT_INTERPTYPE,
    )
    preproc.add_argument(
        '--offsettime',
        dest='offsettime',
        action='store',
        type=float,
        metavar='OFFSETTIME',
        help='Apply offset OFFSETTIME to the lag regressors.',
        default=0.0,
    )
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
    pf.addfilteropts(g_rapidtide, filtertarget='data and regressors', details=True)

    # Add permutation options
    pf.addpermutationopts(g_rapidtide)

    # add window options
    pf.addwindowopts(g_rapidtide, windowtype=DEFAULT_WINDOW_TYPE)

    preproc.add_argument(
        '--detrendorder',
        dest='detrendorder',
        action='store',
        type=int,
        metavar='ORDER',
        help=(f'Set order of trend removal (0 to disable). Default is {DEFAULT_DETREND_ORDER}.'),
        default=DEFAULT_DETREND_ORDER,
    )
    preproc.add_argument(
        '--spatialfilt',
        dest='gausssigma',
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
        '--globalmean',
        dest='useglobalref',
        action='store_true',
        help=(
            'Generate a global mean regressor and use that as the reference '
            'regressor.  If no external regressor is specified, this '
            'is enabled by default.'
        ),
        default=False,
    )
    preproc.add_argument(
        '--globalmaskmethod',
        dest='globalmaskmethod',
        action='store',
        type=str,
        choices=['mean', 'variance'],
        help=(
            'Select whether to use timecourse mean or variance to '
            'mask voxels prior to generating global mean. '
            f'Default is "{DEFAULT_GLOBALMASK_METHOD}".'
        ),
        default=DEFAULT_GLOBALMASK_METHOD,
    )
    preproc.add_argument(
        '--globalmeaninclude',
        dest='globalmeanincludespec',
        metavar='MASK[:VALSPEC]',
        help=(
            'Only use voxels in mask file NAME for global regressor '
            'generation (if VALSPEC is given, only voxels '
            'with integral values listed in VALSPEC are used).'
        ),
        default=None,
    )
    preproc.add_argument(
        '--globalmeanexclude',
        dest='globalmeanexcludespec',
        metavar='MASK[:VALSPEC]',
        help=(
            'Do not use voxels in mask file NAME for global regressor '
            'generation (if VALSPEC is given, only voxels '
            'with integral values listed in VALSPEC are excluded).'
        ),
        default=None,
    )
    preproc.add_argument(
        '--motionfile',
        dest='motionfilespec',
        metavar='MOTFILE',
        help=(
            'Read 6 columns of motion regressors out of MOTFILE file (.par or BIDS .json) '
            '(with timepoints rows) and regress them and/or their derivatives '
            'out of the data prior to analysis. '
        ),
        default=None,
    )
    preproc.add_argument(
        '--motderiv',
        dest='mot_deriv',
        action='store_false',
        help=('Toggle whether derivatives will be used in motion regression.  Default is True.'),
        default=True,
    )
    preproc.add_argument(
        '--confoundfile',
        dest='confoundfilespec',
        metavar='CONFFILE',
        help=(
            'Read additional (non-motion) confound regressors out of CONFFILE file (which can be any type of '
            'multicolumn text file '
            'rapidtide reads as long as data is sampled at TR with timepoints rows).  Optionally do power expansion '
            'and/or calculate derivatives prior to regression. '
        ),
        default=None,
    )
    preproc.add_argument(
        '--confoundpowers',
        dest='confound_power',
        metavar='N',
        type=int,
        help=(
            'Include powers of each confound regressor up to order N. Default is 1 (no expansion). '
        ),
        default=1,
    )
    preproc.add_argument(
        '--confoundderiv',
        dest='confound_deriv',
        action='store_false',
        help=(
            'Toggle whether derivatives will be used in confound regression.  Default is True. '
        ),
        default=True,
    )
    preproc.add_argument(
        '--noconfoundorthogonalize',
        dest='orthogonalize',
        action='store_false',
        help=(
            'Do not orthogonalize confound regressors prior to regressing them out of the data. '
        ),
        default=True,
    )
    preproc.add_argument(
        '--globalsignalmethod',
        dest='globalsignalmethod',
        action='store',
        type=str,
        choices=['sum', 'meanscale', 'pca', 'random'],
        help=(
            'The method for constructing the initial global signal regressor - straight summation, '
            'mean scaling each voxel prior to summation, MLE PCA of the voxels in the global signal mask, '
            'or initializing using random noise.'
            f'Default is "{DEFAULT_GLOBALSIGNAL_METHOD}."'
        ),
        default=DEFAULT_GLOBALSIGNAL_METHOD,
    )
    preproc.add_argument(
        '--globalpcacomponents',
        dest='globalpcacomponents',
        action='store',
        type=float,
        metavar='VALUE',
        help=(
            'Number of PCA components used for estimating the global signal.  If VALUE >= 1, will retain this'
            'many components.  If '
            '0.0 < VALUE < 1.0, enough components will be retained to explain the fraction VALUE of the '
            'total variance. If VALUE is negative, the number of components will be to retain will be selected '
            f'automatically using the MLE method.  Default is {DEFAULT_GLOBAL_PCACOMPONENTS}.'
        ),
        default=DEFAULT_GLOBAL_PCACOMPONENTS,
    )
    preproc.add_argument(
        '--slicetimes',
        dest='slicetimes',
        action='store',
        type=IsFile,
        metavar='FILE',
        help=('Apply offset times from FILE to each slice in the dataset.'),
        default=None,
    )
    preproc.add_argument(
        '--numskip',
        dest='preprocskip',
        action='store',
        type=int,
        metavar='SKIP',
        help=(
            'SKIP TRs were previously deleted during '
            'preprocessing (e.g. if you have done your preprocessing '
            'in FSL and set dummypoints to a nonzero value.) Default is 0. '
        ),
        default=0,
    )
    preproc.add_argument(
        '--numtozero',
        dest='numtozero',
        action='store',
        type=int,
        metavar='NUMPOINTS',
        help=(
            'When calculating the moving regressor, set this number of points to zero at the beginning of the '
            'voxel timecourses. This prevents initial points which may not be in equilibrium from contaminating the '
            'calculated sLFO signal.  This may improve similarity fitting and GLM noise removal.  Default is 0.'
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
            'Limit analysis to data between timepoints '
            'START and END in the fmri file. If END is set to -1, '
            'analysis will go to the last timepoint.  Negative values '
            'of START will be set to 0. Default is to use all timepoints.'
        ),
        default=(-1, -1),
    )
    preproc.add_argument(
        '--nothresh',
        dest='nothresh',
        action='store_true',
        help=('Disable voxel intensity threshold (especially useful for NIRS ' 'data).'),
        default=False,
    )

    # Correlation options
    corr = g_rapidtide.add_argument_group('Correlation options')
    corr.add_argument(
        '--oversampfac',
        dest='oversampfactor',
        action='store',
        type=int,
        metavar='OVERSAMPFAC',
        help=(
            'Oversample the fMRI data by the following '
            'integral factor.  Set to -1 for automatic selection (default).'
        ),
        default=-1,
    )
    corr.add_argument(
        '--regressor',
        dest='regressorfile',
        action='store',
        type=IsFile,
        metavar='FILE',
        help=(
            'Read the initial probe regressor from file FILE (if not '
            'specified, generate and use the global regressor).'
        ),
        default=None,
    )

    reg_group = corr.add_mutually_exclusive_group()
    reg_group.add_argument(
        '--regressorfreq',
        dest='inputfreq',
        action='store',
        type=lambda x: pf.is_float(parser, x),
        metavar='FREQ',
        help=(
            'Probe regressor in file has sample '
            'frequency FREQ (default is 1/tr) '
            'NB: --regressorfreq and --regressortstep) '
            'are two ways to specify the same thing.'
        ),
        default='auto',
    )
    reg_group.add_argument(
        '--regressortstep',
        dest='inputfreq',
        action='store',
        type=lambda x: pf.invert_float(parser, x),
        metavar='TSTEP',
        help=(
            'Probe regressor in file has sample '
            'frequency FREQ (default is 1/tr) '
            'NB: --regressorfreq and --regressortstep) '
            'are two ways to specify the same thing.'
        ),
        default='auto',
    )

    corr.add_argument(
        '--regressorstart',
        dest='inputstarttime',
        action='store',
        type=float,
        metavar='START',
        help=(
            'The time delay in seconds into the regressor '
            'file, corresponding in the first TR of the fMRI '
            'file (default is 0.0).'
        ),
        default=None,
    )
    corr.add_argument(
        '--corrweighting',
        dest='corrweighting',
        action='store',
        type=str,
        choices=['None', 'phat', 'liang', 'eckart', 'regressor'],
        help=(
            "Method to use for cross-correlation weighting. "
            "'None' performs an unweighted correlation. "
            "'phat' weights the correlation by the magnitude of the product of the timecourse's FFTs. "
            "'liang' weights the correlation by the sum of the magnitudes of the timecourse's FFTs. "
            "'eckart' weights the correlation by the product of the magnitudes of the timecourse's FFTs. "
            "'regressor' weights the correlation by the magnitude of the sLFO regressor FFT. "
            f'Default is "{DEFAULT_CORRWEIGHTING}".'
        ),
        default=DEFAULT_CORRWEIGHTING,
    )
    corr.add_argument(
        '--corrtype',
        dest='corrtype',
        action='store',
        type=str,
        choices=['linear', 'circular'],
        help=('Cross-correlation type (linear or circular). ' f'Default is "{DEFAULT_CORRTYPE}".'),
        default=DEFAULT_CORRTYPE,
    )

    mask_group = corr.add_mutually_exclusive_group()
    mask_group.add_argument(
        '--corrmaskthresh',
        dest='corrmaskthreshpct',
        action='store',
        type=float,
        metavar='PCT',
        help=(
            'Do correlations in voxels where the mean '
            'exceeds this percentage of the robust max. '
            f'Default is {DEFAULT_CORRMASK_THRESHPCT}. '
        ),
        default=DEFAULT_CORRMASK_THRESHPCT,
    )
    mask_group.add_argument(
        '--corrmask',
        dest='corrmaskincludespec',
        metavar='MASK[:VALSPEC]',
        help=(
            'Only do correlations in nonzero voxels in NAME '
            '(if VALSPEC is given, only voxels '
            'with integral values listed in VALSPEC are used). '
        ),
        default=None,
    )
    corr.add_argument(
        '--similaritymetric',
        dest='similaritymetric',
        action='store',
        type=str,
        choices=['correlation', 'mutualinfo', 'hybrid'],
        help=(
            'Similarity metric for finding delay values.  '
            'Choices are "correlation", "mutualinfo", and "hybrid". '
            f'Default is {DEFAULT_SIMILARITYMETRIC}.'
        ),
        default=DEFAULT_SIMILARITYMETRIC,
    )
    corr.add_argument(
        '--mutualinfosmoothingtime',
        dest='smoothingtime',
        action='store',
        type=float,
        metavar='TAU',
        help=(
            'Time constant of a temporal smoothing function to apply to the '
            'mutual information function. '
            f'Default is {DEFAULT_MUTUALINFO_SMOOTHINGTIME} seconds.  '
            'TAU <=0.0 disables smoothing.'
        ),
        default=DEFAULT_MUTUALINFO_SMOOTHINGTIME,
    )
    corr.add_argument(
        '--simcalcrange',
        dest='simcalcrange',
        action='store',
        nargs=2,
        type=int,
        metavar=('START', 'END'),
        help=(
            "Limit correlation calculation to data between timepoints "
            "START and END in the fmri file. If END is set to -1, "
            "analysis will go to the last timepoint.  Negative values "
            "of START will be set to 0. Default is to use all timepoints. "
            "NOTE: these offsets are relative to the start of the "
            "dataset AFTER any trimming done with '--timerange'."
        ),
        default=(-1, -1),
    )

    # Correlation fitting options
    corr_fit = g_rapidtide.add_argument_group('Correlation fitting options')

    fixdelay = corr_fit.add_mutually_exclusive_group()
    fixdelay.add_argument(
        '--fixdelay',
        dest='fixeddelayvalue',
        action='store',
        type=float,
        metavar='DELAYTIME',
        help=("Don't fit the delay time - set it to DELAYTIME seconds for all " "voxels."),
        default=None,
    )
    fixdelay.add_argument(
        '--searchrange',
        dest='lag_extrema',
        action=pf.IndicateSpecifiedAction,
        nargs=2,
        type=float,
        metavar=('LAGMIN', 'LAGMAX'),
        help='Limit fit to a range of lags from LAGMIN to LAGMAX.',
        default=(DEFAULT_LAGMIN, DEFAULT_LAGMAX),
    )
    corr_fit.add_argument(
        '--sigmalimit',
        dest='widthmax',
        action='store',
        type=float,
        metavar='SIGMALIMIT',
        help=(
            'Reject lag fits with linewidth wider than '
            f'SIGMALIMIT Hz. Default is {DEFAULT_SIGMAMAX} Hz.'
        ),
        default=DEFAULT_SIGMAMAX,
    )
    corr_fit.add_argument(
        '--bipolar',
        dest='bipolar',
        action='store_true',
        help=('Bipolar mode - match peak correlation ignoring sign.'),
        default=False,
    )
    corr_fit.add_argument(
        '--nofitfilt',
        dest='zerooutbadfit',
        action='store_false',
        help=('Do not zero out peak fit values if fit fails.'),
        default=True,
    )
    corr_fit.add_argument(
        '--peakfittype',
        dest='peakfittype',
        action='store',
        type=str,
        choices=['gauss', 'fastgauss', 'quad', 'fastquad', 'COM', 'None'],
        help=(
            'Method for fitting the peak of the similarity function '
            '"gauss" performs a Gaussian fit, and is most accurate. '
            '"quad" and "fastquad" use a quadratic fit, '
            'which is faster, but not as well tested. '
            f'Default is "{DEFAULT_PEAKFIT_TYPE}".'
        ),
        default=DEFAULT_PEAKFIT_TYPE,
    )
    corr_fit.add_argument(
        '--despecklepasses',
        dest='despeckle_passes',
        action=pf.IndicateSpecifiedAction,
        type=int,
        metavar='PASSES',
        help=(
            'Detect and refit suspect correlations to '
            'disambiguate peak locations in PASSES '
            f'passes.  Default is to perform {DEFAULT_DESPECKLE_PASSES} passes. '
            'Set to 0 to disable.'
        ),
        default=DEFAULT_DESPECKLE_PASSES,
    )
    corr_fit.add_argument(
        '--despecklethresh',
        dest='despeckle_thresh',
        action='store',
        type=float,
        metavar='VAL',
        help=(
            'Refit correlation if median discontinuity '
            'magnitude exceeds VAL. '
            f'Default is {DEFAULT_DESPECKLE_THRESH} seconds.'
        ),
        default=DEFAULT_DESPECKLE_THRESH,
    )

    # Regressor refinement options
    reg_ref = g_rapidtide.add_argument_group('Regressor refinement options')
    reg_ref.add_argument(
        '--refineprenorm',
        dest='refineprenorm',
        action='store',
        type=str,
        choices=['None', 'mean', 'var', 'std', 'invlag'],
        help=(
            'Apply TYPE prenormalization to each '
            'timecourse prior to refinement. '
            f'Default is "{DEFAULT_REFINE_PRENORM}".'
        ),
        default=DEFAULT_REFINE_PRENORM,
    )
    reg_ref.add_argument(
        '--refineweighting',
        dest='refineweighting',
        action='store',
        type=str,
        choices=['None', 'NIRS', 'R', 'R2'],
        help=(
            'Apply TYPE weighting to each timecourse prior '
            f'to refinement. Default is "{DEFAULT_REFINE_WEIGHTING}".'
        ),
        default=DEFAULT_REFINE_WEIGHTING,
    )
    reg_ref.add_argument(
        '--passes',
        dest='passes',
        action='store',
        type=int,
        metavar='PASSES',
        help=('Set the number of processing passes to PASSES.  ' f'Default is {DEFAULT_PASSES}.'),
        default=DEFAULT_PASSES,
    )
    reg_ref.add_argument(
        '--refineinclude',
        dest='refineincludespec',
        metavar='MASK[:VALSPEC]',
        help=(
            'Only use voxels in file MASK for regressor refinement '
            '(if VALSPEC is given, only voxels '
            'with integral values listed in VALSPEC are used). '
        ),
        default=None,
    )
    reg_ref.add_argument(
        '--refineexclude',
        dest='refineexcludespec',
        metavar='MASK[:VALSPEC]',
        help=(
            'Do not use voxels in file MASK for regressor refinement '
            '(if VALSPEC is given, voxels '
            'with integral values listed in VALSPEC are excluded). '
        ),
        default=None,
    )
    reg_ref.add_argument(
        '--norefinedespeckled',
        dest='refinedespeckled',
        action='store_false',
        help=('Do not use despeckled pixels in calculating the refined regressor.'),
        default=True,
    )
    reg_ref.add_argument(
        '--lagminthresh',
        dest='lagminthresh',
        action='store',
        metavar='MIN',
        type=float,
        help=(
            'For refinement, exclude voxels with delays '
            f'less than MIN. Default is {DEFAULT_LAGMIN_THRESH} seconds. '
        ),
        default=DEFAULT_LAGMIN_THRESH,
    )
    reg_ref.add_argument(
        '--lagmaxthresh',
        dest='lagmaxthresh',
        action='store',
        metavar='MAX',
        type=float,
        help=(
            'For refinement, exclude voxels with delays '
            f'greater than MAX. Default is {DEFAULT_LAGMAX_THRESH} seconds. '
        ),
        default=DEFAULT_LAGMAX_THRESH,
    )
    reg_ref.add_argument(
        '--ampthresh',
        dest='ampthresh',
        action='store',
        metavar='AMP',
        type=float,
        help=(
            'For refinement, exclude voxels with correlation '
            f'coefficients less than AMP (default is {DEFAULT_AMPTHRESH}).  '
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
        help=(
            'For refinement, exclude voxels with widths '
            f'greater than SIGMA seconds. Default is {DEFAULT_SIGMATHRESH} seconds.'
        ),
        default=DEFAULT_SIGMATHRESH,
    )
    reg_ref.add_argument(
        '--offsetinclude',
        dest='offsetincludespec',
        metavar='MASK[:VALSPEC]',
        help=(
            'Only use voxels in file MASK for determining the zero time offset value '
            '(if VALSPEC is given, only voxels '
            'with integral values listed in VALSPEC are used). '
        ),
        default=None,
    )
    reg_ref.add_argument(
        '--offsetexclude',
        dest='offsetexcludespec',
        metavar='MASK[:VALSPEC]',
        help=(
            'Do not use voxels in file MASK for determining the zero time offset value '
            '(if VALSPEC is given, voxels '
            'with integral values listed in VALSPEC are excluded). '
        ),
        default=None,
    )
    reg_ref.add_argument(
        '--norefineoffset',
        dest='refineoffset',
        action='store_false',
        help=('Disable realigning refined regressor to zero lag.'),
        default=True,
    )
    reg_ref.add_argument(
        '--nopickleft',
        dest='pickleft',
        action='store_false',
        help=('Disables selecting the leftmost delay peak when setting the refine offset.'),
        default=True,
    )
    reg_ref.add_argument(
        '--pickleft',
        dest='dummy',
        action='store_true',
        help=(
            "DEPRECATED. pickleft is now on by default. Use 'nopickleft' to disable it instead."
        ),
        default=True,
    )
    reg_ref.add_argument(
        '--pickleftthresh',
        dest='pickleftthresh',
        action='store',
        metavar='THRESH',
        type=float,
        help=(
            'Threshhold value (fraction of maximum) in a histogram '
            f'to be considered the start of a peak.  Default is {DEFAULT_PICKLEFT_THRESH}.'
        ),
        default=DEFAULT_PICKLEFT_THRESH,
    )

    refine = reg_ref.add_mutually_exclusive_group()
    refine.add_argument(
        '--refineupperlag',
        dest='lagmaskside',
        action='store_const',
        const='upper',
        help=('Only use positive lags for regressor refinement.'),
        default='both',
    )
    refine.add_argument(
        '--refinelowerlag',
        dest='lagmaskside',
        action='store_const',
        const='lower',
        help=('Only use negative lags for regressor refinement.'),
        default='both',
    )
    reg_ref.add_argument(
        '--refinetype',
        dest='refinetype',
        action='store',
        type=str,
        choices=['pca', 'ica', 'weighted_average', 'unweighted_average'],
        help=(
            'Method with which to derive refined regressor. '
            f'Default is "{DEFAULT_REFINE_TYPE}".'
        ),
        default=DEFAULT_REFINE_TYPE,
    )
    reg_ref.add_argument(
        '--pcacomponents',
        dest='pcacomponents',
        action='store',
        type=float,
        metavar='VALUE',
        help=(
            'Number of PCA components used for refinement.  If VALUE >= 1, will retain this many components.  If '
            '0.0 < VALUE < 1.0, enough components will be retained to explain the fraction VALUE of the '
            'total variance. If VALUE is negative, the number of components will be to retain will be selected '
            f'automatically using the MLE method.  Default is {DEFAULT_REFINE_PCACOMPONENTS}.'
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
            'By default, this is not set, so refinement will run for the specified number of passes. '
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
            'Terminate refinement after MAXPASSES passes, whether or not convergence has occured. '
            f'Default is {DEFAULT_MAXPASSES}.'
        ),
        default=DEFAULT_MAXPASSES,
    )

    # GLM noise removal options
    glm = g_rapidtide.add_argument_group('GLM noise removal options')
    glm.add_argument(
        '--noglm',
        dest='doglmfilt',
        action='store_false',
        help=(
            'Turn off GLM filtering to remove delayed '
            'regressor from each voxel (disables output of '
            'fitNorm).'
        ),
        default=True,
    )
    glm.add_argument(
        '--glmsourcefile',
        dest='glmsourcefile',
        action='store',
        type=IsFile,
        metavar='FILE',
        help=(
            'Regress delayed regressors out of FILE instead '
            'of the initial fmri file used to estimate '
            'delays.'
        ),
        default=None,
    )
    glm.add_argument(
        '--preservefiltering',
        dest='preservefiltering',
        action='store_true',
        help="Don't reread data prior to performing GLM.",
        default=False,
    )
    glm.add_argument(
        '--glmderivs',
        dest='glmderivs',
        action='store',
        type=int,
        metavar='NDERIVS',
        help=(
            f'When doing final GLM, include derivatives up to NDERIVS order. Default is {DEFAULT_GLMDERIVS}'
        ),
        default=DEFAULT_GLMDERIVS,
    )

    # Output options
    output = g_rapidtide.add_argument_group('Output options')
    output.add_argument(
        '--outputlevel',
        dest='outputlevel',
        action='store',
        type=str,
        choices=['min', 'less', 'normal', 'more', 'max'],
        help=(
            "The level of file output produced.  "
            "'min' produces only absolutely essential files, "
            "'less' adds in the GLM filtered data (rather than just filter efficacy metrics), "
            "'normal' saves what you would typically want around for interactive data "
            "exploration, "
            "'more' adds files that are sometimes useful, and 'max' outputs anything you "
            "might possibly want. "
            "Selecting 'max' will produce ~3x your input datafile size as output."
        ),
        default=DEFAULT_OUTPUTLEVEL,
    )
    output.add_argument(
        '--nolimitoutput',
        dest='limitoutput',
        action='store_false',
        help=(
            "Save some of the large and rarely used files.  "
            "NB: THIS IS NOW DEPRECATED: Use '--outputlevel max' instead."
        ),
        default=True,
    )
    output.add_argument(
        '--savelags',
        dest='savecorrtimes',
        action='store_true',
        help='Save a table of lagtimes used.',
        default=False,
    )
    output.add_argument(
        '--histlen',  # was -h
        dest='histlen',
        action='store',
        type=int,
        metavar='HISTLEN',
        help='Change the histogram length to HISTLEN.',
        default=DEFAULT_HISTLEN,
    )
    output.add_argument(
        '--saveintermediatemaps',
        dest='saveintermediatemaps',
        action='store_true',
        help='Save lag times, strengths, widths, and mask for each pass.',
        default=False,
    )
    output.add_argument(
        '--calccoherence',
        dest='calccoherence',
        action='store_true',
        help=('Calculate and save the coherence between the final regressor and the data.'),
        default=False,
    )

    # Add version options
    pf.addversionopts(g_rapidtide)

    # Performance options
    perf = g_rapidtide.add_argument_group('Performance options')
    perf.add_argument(
        '--nprocs',
        dest='nprocs',
        action='store',
        type=int,
        metavar='NPROCS',
        help=(
            'Use NPROCS worker processes for multiprocessing. '
            'Setting NPROCS to less than 1 sets the number of '
            'worker processes to n_cpus (unless --reservecpu is used).'
        ),
        default=1,
    )
    perf.add_argument(
        '--reservecpu',
        dest='reservecpu',
        action='store_true',
        help=(
            'When automatically setting nprocs, reserve one CPU for '
            'process management rather than using them all for worker threads.'
        ),
        default=False,
    )
    perf.add_argument(
        '--mklthreads',
        dest='mklthreads',
        action='store',
        type=int,
        metavar='MKLTHREADS',
        help=(
            'If mkl library is installed, use no more than MKLTHREADS worker '
            'threads in accelerated numpy calls.  Set to -1 to use the maximum available. '
            'Default is 1.'
        ),
        default=1,
    )
    perf.add_argument(
        '--nonumba',
        dest='nonumba',
        action='store_true',
        help=(
            'By default, numba is used if present.  Use this option to disable jit '
            'compilation with numba even if it is installed.'
        ),
        default=False,
    )

    # Miscellaneous options
    misc = g_rapidtide.add_argument_group('Miscellaneous options')
    misc.add_argument(
        '--noprogressbar',
        dest='showprogressbar',
        action='store_false',
        help=('Will disable showing progress bars (helpful if stdout is going to a file).'),
        default=True,
    )
    misc.add_argument(
        '--checkpoint',
        dest='checkpoint',
        action='store_true',
        help='Enable run checkpoints.',
        default=False,
    )
    misc.add_argument(
        '--spcalculation',
        dest='internalprecision',
        action='store_const',
        const='single',
        help=(
            'Use single precision for internal calculations '
            '(may be useful when RAM is limited).'
        ),
        default='double',
    )
    misc.add_argument(
        '--dpoutput',
        dest='outputprecision',
        action='store_const',
        const='double',
        help=('Use double precision for output files.'),
        default='single',
    )
    misc.add_argument(
        '--cifti',
        dest='isgrayordinate',
        action='store_true',
        help='Data file is a converted CIFTI.',
        default=False,
    )
    misc.add_argument(
        '--simulate',
        dest='fakerun',
        action='store_true',
        help='Simulate a run - just report command line options.',
        default=False,
    )
    misc.add_argument(
        '--displayplots',
        dest='displayplots',
        action='store_true',
        help='Display plots of interesting timecourses.',
        default=False,
    )
    misc.add_argument(
        '--nosharedmem',
        dest='sharedmem',
        action='store_false',
        help=('Disable use of shared memory for large array storage.'),
        default=True,
    )
    misc.add_argument(
        '--memprofile',
        dest='memprofile',
        action='store_true',
        help=('Enable memory profiling - ' 'warning: this slows things down a lot.'),
        default=False,
    )
    pf.addtagopts(misc)

    # Experimental options (not fully tested, may not work)
    experimental = g_rapidtide.add_argument_group(
        'Experimental options (not fully tested, or not tested at all, may not work).  Beware!'
    )
    experimental.add_argument(
        '--territorymap',
        dest='territorymap',
        metavar='MAP[:VALSPEC]',
        help=(
            "This specifies a territory map.  Each territory is a set of voxels with the same integral value.  "
            "If VALSPEC is given, only territories in the mask with integral values listed in VALSPEC are used, otherwise "
            "all nonzero voxels are used.  If this option is set, certain output measures will be summarized over "
            "each territory in the map, in addition to over the whole brain.  Some interesting territory maps might be: "
            "a gray/white/csf segmentation image, an arterial territory map, lesion area vs. healthy "
            "tissue segmentation, etc.  NB: at the moment this is just a placeholder - it doesn't do anything."
        ),
        default=None,
    )
    experimental.add_argument(
        '--psdfilter',
        dest='psdfilter',
        action='store_true',
        help=('Apply a PSD weighted Wiener filter to shifted timecourses prior to refinement.'),
        default=False,
    )
    experimental.add_argument(
        '--wiener',
        dest='dodeconv',
        action='store_true',
        help=('Do Wiener deconvolution to find voxel transfer function.'),
        default=False,
    )
    experimental.add_argument(
        '--corrbaselinespatialsigma',
        dest='corrbaselinespatialsigma',
        action='store',
        type=float,
        metavar='SIGMA',
        help=('Spatial lowpass kernel, in mm, for filtering the correlation function baseline. '),
        default=0.0,
    )
    experimental.add_argument(
        '--corrbaselinetemphpfcutoff',
        dest='corrbaselinetemphpfcutoff',
        action='store',
        type=float,
        metavar='FREQ',
        help=(
            'Temporal highpass cutoff, in Hz, for filtering the correlation function baseline. '
        ),
        default=0.0,
    )
    experimental.add_argument(
        '--spatialtolerance',
        dest='spatialtolerance',
        action='store',
        type=float,
        metavar='EPSILON',
        help=(
            'When checking to see if the spatial dimensions of two NIFTI files match, allow a relative difference '
            'of EPSILON in any dimension.  By default, this is set to 0.0, requiring an exact match. '
        ),
        default=0.0,
    )
    experimental.add_argument(
        '--echocancel',
        dest='echocancel',
        action='store_true',
        help=('Attempt to perform echo cancellation on current moving regressor.'),
        default=False,
    )
    experimental.add_argument(
        '--autorespdelete',
        dest='respdelete',
        action='store_true',
        help=('Attempt to detect and remove respiratory signal that strays into ' 'the LFO band.'),
        default=False,
    )

    experimental.add_argument(
        '--noisetimecourse',
        dest='noisetimecoursespec',
        metavar='FILENAME[:VALSPEC]',
        help=(
            "Find and remove any instance of the timecourse supplied from any regressors used for analysis. "
            "(if VALSPEC is given, and there are multiple timecourses in the file, use the indicated timecourse."
            "This can be the name of the regressor if it's in the file, or the column number). "
        ),
        default=None,
    )
    noise_group = experimental.add_mutually_exclusive_group()
    noise_group.add_argument(
        '--noisefreq',
        dest='noisefreq',
        action='store',
        type=lambda x: pf.is_float(parser, x),
        metavar='FREQ',
        help=(
            'Noise timecourse in file has sample '
            'frequency FREQ (default is 1/tr) '
            'NB: --noisefreq and --noisetstep) '
            'are two ways to specify the same thing.'
        ),
        default='auto',
    )
    noise_group.add_argument(
        '--noisetstep',
        dest='noisefreq',
        action='store',
        type=lambda x: pf.invert_float(parser, x),
        metavar='TSTEP',
        help=(
            'Noise timecourse in file has sample '
            'frequency FREQ (default is 1/tr) '
            'NB: --noisefreq and --noisetstep) '
            'are two ways to specify the same thing.'
        ),
        default='auto',
    )
    experimental.add_argument(
        '--noisestart',
        dest='noisestarttime',
        action='store',
        type=float,
        metavar='START',
        help=(
            'The time delay in seconds into the noise timecourse '
            'file, corresponding in the first TR of the fMRI '
            'file (default is 0.0).'
        ),
        default=0.0,
    )
    experimental.add_argument(
        '--noiseinvert',
        dest='noiseinvert',
        action='store_true',
        help=('Invert noise regressor prior to alignment.'),
        default=False,
    )

    experimental.add_argument(
        '--acfix',
        dest='fix_autocorrelation',
        action='store_true',
        help=(
            'Check probe regressor for autocorrelations in order to disambiguate peak location.'
        ),
        default=False,
    )
    experimental.add_argument(
        '--negativegradient',
        dest='negativegradient',
        action='store_true',
        help=(
            'Calculate the negative gradient of the fmri data after spectral filtering '
            'so you can look for CSF flow à la '
            'https://www.biorxiv.org/content/10.1101/2021.03.29.437406v1.full. '
        ),
        default=False,
    )
    experimental.add_argument(
        '--negativegradregressor',
        dest='negativegradregressor',
        action='store_true',
        help=argparse.SUPPRESS,
        default=False,
    )
    experimental.add_argument(
        '--cleanrefined',
        dest='cleanrefined',
        action='store_true',
        help=(
            'Perform additional processing on refined '
            'regressor to remove spurious '
            'components.'
        ),
        default=False,
    )
    experimental.add_argument(
        '--dispersioncalc',
        dest='dodispersioncalc',
        action='store_true',
        help=('Generate extra data during refinement to allow calculation of dispersion.'),
        default=False,
    )
    experimental.add_argument(
        '--tincludemask',
        dest='tincludemaskname',
        action='store',
        type=IsFile,
        metavar='FILE',
        help=(
            'Only correlate during epochs specified '
            'in MASKFILE (NB: each line of FILE '
            'contains the time and duration of an '
            'epoch to include.'
        ),
        default=None,
    )
    experimental.add_argument(
        '--texcludemask',
        dest='texcludemaskname',
        action='store',
        type=IsFile,
        metavar='FILE',
        help=(
            'Do not correlate during epochs specified '
            'in MASKFILE (NB: each line of FILE '
            'contains the time and duration of an '
            'epoch to exclude.'
        ),
        default=None,
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
        import pkgutil

        if pkgutil.find_loader('sentry_sdk') is None:
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
            "The selected output folder is the same as the input BIDS folder. "
            "Please modify the output path "
            f"(suggestion: {bids_dir / 'derivatives' / 'fmripost_rapidtide-' + version.split('+')[0]}."
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
            'Making sure the input data is BIDS compliant '
            '(warnings can be ignored in most cases).'
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
            "One or more participant labels were not found in the BIDS directory: "
            f"{', '.join(missing_subjects)}."
        )

    config.execution.participant_label = sorted(participant_label)
