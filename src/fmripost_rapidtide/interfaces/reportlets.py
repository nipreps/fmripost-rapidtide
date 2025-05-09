# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
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
"""ReportCapableInterfaces for segmentation tools."""

import os
import re
import time
from collections import Counter
from uuid import uuid4

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiObject,
    SimpleInterface,
    Str,
    TraitedSpec,
    isdefined,
    traits,
)
from nireports._vendored.svgutils.transform import fromstring
from nireports.reportlets.utils import compose_view, cuts_from_bbox, extract_svg

SUBJECT_TEMPLATE = """\
\t<ul class="elem-desc">
\t\t<li>Subject ID: {subject_id}</li>
\t\t<li>Functional series: {n_bold:d}</li>
{tasks}
\t\t<li>Standard output spaces: {std_spaces}</li>
\t\t<li>Non-standard output spaces: {nstd_spaces}</li>
\t</ul>
"""

FUNCTIONAL_TEMPLATE = """\
\t\t<details open>
\t\t<summary>Summary</summary>
\t\t<ul class="elem-desc">
\t\t\t<li>Original orientation: {ornt}</li>
\t\t\t<li>Repetition time (TR): {tr:.03g}s</li>
\t\t\t<li>Phase-encoding (PE) direction: {pedir}</li>
\t\t\t<li>Slice timing correction: {stc}</li>
\t\t\t<li>Susceptibility distortion correction: {sdc}</li>
\t\t\t<li>Registration: {registration}</li>
\t\t\t<li>Non-steady-state volumes: {dummy_scan_desc}</li>
\t\t</ul>
\t\t</details>
"""

ABOUT_TEMPLATE = """\t<ul>
\t\t<li>fMRIPost-rapidtide version: {version}</li>
\t\t<li>fMRIPost-rapidtide command: <code>{command}</code></li>
\t\t<li>Date preprocessed: {date}</li>
\t</ul>
</div>
"""


class SummaryOutputSpec(TraitedSpec):
    out_report = File(exists=True, desc='HTML segment containing summary')


class SummaryInterface(SimpleInterface):
    output_spec = SummaryOutputSpec

    def _run_interface(self, runtime):
        segment = self._generate_segment()
        fname = os.path.join(runtime.cwd, 'report.html')
        with open(fname, 'w') as fobj:
            fobj.write(segment)
        self._results['out_report'] = fname
        return runtime

    def _generate_segment(self):
        raise NotImplementedError


class SubjectSummaryInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(desc='Subject ID')
    bold = InputMultiObject(
        traits.Either(File(exists=True), traits.List(File(exists=True))),
        desc='BOLD functional series',
    )
    std_spaces = traits.List(Str, desc='list of standard spaces')
    nstd_spaces = traits.List(Str, desc='list of non-standard spaces')


class SubjectSummary(SummaryInterface):
    input_spec = SubjectSummaryInputSpec
    output_spec = SummaryOutputSpec

    def _generate_segment(self):
        BIDS_NAME = re.compile(
            r'^(.*\/)?'
            '(?P<subject_id>sub-[a-zA-Z0-9]+)'
            '(_(?P<session_id>ses-[a-zA-Z0-9]+))?'
            '(_(?P<task_id>task-[a-zA-Z0-9]+))?'
            '(_(?P<acq_id>acq-[a-zA-Z0-9]+))?'
            '(_(?P<rec_id>rec-[a-zA-Z0-9]+))?'
            '(_(?P<run_id>run-[a-zA-Z0-9]+))?'
        )

        # Add list of tasks with number of runs
        bold_series = self.inputs.bold if isdefined(self.inputs.bold) else []
        bold_series = [s[0] if isinstance(s, list) else s for s in bold_series]

        counts = Counter(
            BIDS_NAME.search(series).groupdict()['task_id'][5:] for series in bold_series
        )

        tasks = ''
        if counts:
            header = '\t\t<ul class="elem-desc">'
            footer = '\t\t</ul>'
            lines = [
                '\t\t\t<li>Task: {task_id} ({n_runs:d} run{s})</li>'.format(
                    task_id=task_id, n_runs=n_runs, s='' if n_runs == 1 else 's'
                )
                for task_id, n_runs in sorted(counts.items())
            ]
            tasks = '\n'.join([header] + lines + [footer])

        return SUBJECT_TEMPLATE.format(
            subject_id=self.inputs.subject_id,
            n_bold=len(bold_series),
            tasks=tasks,
            std_spaces=', '.join(self.inputs.std_spaces),
            nstd_spaces=', '.join(self.inputs.nstd_spaces),
        )


class AboutSummaryInputSpec(BaseInterfaceInputSpec):
    version = Str(desc='FMRIPREP version')
    command = Str(desc='FMRIPREP command')
    # Date not included - update timestamp only if version or command changes


class AboutSummary(SummaryInterface):
    input_spec = AboutSummaryInputSpec

    def _generate_segment(self):
        return ABOUT_TEMPLATE.format(
            version=self.inputs.version,
            command=self.inputs.command,
            date=time.strftime('%Y-%m-%d %H:%M:%S %z'),
        )


class _FCInflationPlotInputSpecRPT(BaseInterfaceInputSpec):
    fcinflation_file = File(
        exists=True,
        mandatory=True,
        desc='FC inflation time series',
    )
    out_report = File(
        'fcinflation_reportlet.svg',
        usedefault=True,
        desc='Filename for the visual report generated by Nipype.',
    )


class _FCInflationPlotOutputSpecRPT(TraitedSpec):
    out_report = File(
        exists=True,
        desc='Filename for the visual report generated by Nipype.',
    )


class FCInflationPlotRPT(SimpleInterface):
    """Create a reportlet for Rapidtide outputs."""

    input_spec = _FCInflationPlotInputSpecRPT
    output_spec = _FCInflationPlotOutputSpecRPT

    def _run_interface(self, runtime):
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns

        sns.set_theme(style='whitegrid')

        out_file = os.path.abspath(self.inputs.out_report)

        df = pd.read_table(self.inputs.fcinflation_file)
        df['timepoint'] = np.arange(df.shape[0])

        fig, ax = plt.subplots(figsize=(16, 8))
        palette = ['red', 'lightblue', 'blue']
        for i_col, col in enumerate(['preprocessed', 'denoised']):
            df[f'{col}_mean_minus_std'] = df[f'{col}_mean'] - df[f'{col}_std']
            df[f'{col}_mean_plus_std'] = df[f'{col}_mean'] + df[f'{col}_std']
            sns.lineplot(x='timepoint', y=f'{col}_mean', data=df, ax=ax, color=palette[i_col])
            ax.fill_between(
                x=df['timepoint'],
                y1=df[f'{col}_mean_minus_std'],
                y2=df[f'{col}_mean_plus_std'],
                alpha=0.5,
                color=palette[i_col],
            )
        ax.set_xlim(0, df['timepoint'].max())
        fig.savefig(out_file)
        self._results['out_report'] = out_file
        return runtime


class _StatisticalMapInputSpecRPT(BaseInterfaceInputSpec):
    overlay = File(
        exists=True,
        mandatory=True,
        desc='FC inflation time series',
    )
    underlay = File(
        exists=True,
        mandatory=True,
        desc='Underlay image',
    )
    mask = File(
        exists=True,
        mandatory=False,
        desc='Mask image',
    )
    cmap = traits.Str(
        'viridis',
        desc='Colormap',
        usedefault=True,
    )
    out_report = File(
        'statistical_map_report.svg',
        usedefault=True,
        desc='Filename for the visual report generated by Nipype.',
    )


class _StatisticalMapOutputSpecRPT(TraitedSpec):
    out_report = File(
        exists=True,
        desc='Filename for the visual report generated by Nipype.',
    )


class StatisticalMapRPT(SimpleInterface):
    """Create a reportlet for Rapidtide outputs."""

    input_spec = _StatisticalMapInputSpecRPT
    output_spec = _StatisticalMapOutputSpecRPT

    def _run_interface(self, runtime):
        from nilearn import image, masking, plotting

        out_file = os.path.abspath(self.inputs.out_report)

        if isdefined(self.inputs.mask):
            mask_img = image.load_img(self.inputs.mask)
            overlay_img = masking.unmask(
                masking.apply_mask(self.inputs.overlay, self.inputs.mask),
                self.inputs.mask,
            )
            # since the moving image is already in the fixed image space we
            # should apply the same mask
            underlay_img = image.load_img(self.inputs.underlay)
        else:
            overlay_img = image.load_img(self.inputs.overlay)
            underlay_img = image.load_img(self.inputs.underlay)
            mask_img = image.threshold_img(overlay_img, 1e-3)

        n_cuts = 7
        cuts = cuts_from_bbox(mask_img, cuts=n_cuts)
        order = ('z', 'x', 'y')
        out_files = []

        # Plot each cut axis
        plot_params = {}
        for mode in list(order):
            plot_params['display_mode'] = mode
            plot_params['cut_coords'] = cuts[mode]
            plot_params['title'] = None
            plot_params['cmap'] = self.inputs.cmap

            # Generate nilearn figure
            display = plotting.plot_stat_map(
                overlay_img,
                bg_img=underlay_img,
                **plot_params,
            )

            svg = extract_svg(display, compress=False)
            display.close()

            # Find and replace the figure_1 id.
            svg = svg.replace('figure_1', f'{mode}-{uuid4()}', 1)
            out_files.append(fromstring(svg))

        compose_view(bg_svgs=out_files, fg_svgs=None, out_file=out_file)
        self._results['out_report'] = out_file
        return runtime
