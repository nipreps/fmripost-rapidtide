"""Interfaces to calculate confounds."""

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)


class _FCInflationInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        desc='4D BOLD file in MNI152NLin6Asym:res-2mm space.',
    )
    TR = traits.Float(
        mandatory=False,
        desc='Repetition time, in seconds. Will be inferred from the BOLD file if not provided.',
    )
    mask = File(
        exists=True,
        mandatory=True,
        desc='Brain mask in same space+resolution as in_file.',
    )
    window = traits.Float(
        21.6,
        usedefault=True,
        desc='Window length in seconds (not TRs) to compute FC inflation.',
    )


class _FCInflationOutputSpec(TraitedSpec):
    fc_inflation = File(exists=True, desc='FC inflation TSV file.')
    metrics = traits.Dict(desc='Metrics summarizing FC inflation.')


class FCInflation(SimpleInterface):
    """Compute functional connectivity inflation in a 4D BOLD file."""

    input_spec = _FCInflationInputSpec
    output_spec = _FCInflationOutputSpec

    def _run_interface(self, runtime):
        import os

        import nibabel as nb
        import numpy as np
        import pandas as pd
        from nilearn.maskers import NiftiLabelsMasker
        from templateflow.api import get as get_template

        atlas = str(
            get_template(
                'MNI152NLin6Asym',
                resolution='02',
                atlas='Schaefer2018',
                desc='100Parcels7Networks',
                extension='nii.gz',
            )
        )
        masker = NiftiLabelsMasker(labels_img=atlas, mask_img=self.inputs.mask)
        data = masker.fit_transform(self.inputs.in_file)
        n_vols, n_voxels = data.shape
        t_r = self.inputs.TR
        if not isdefined(t_r):
            t_r = nb.load(self.inputs.in_file).header.get_zooms()[3]

        window_t_r = int(np.ceil(self.inputs.window / t_r))

        if n_vols <= window_t_r:
            raise ValueError(
                f'Number of volumes ({n_vols}) must be greater than window length ({window_t_r}).'
            )

        idx = np.triu_indices(n_voxels, k=1)
        out_dfs = []
        for i in range(n_vols - window_t_r):
            sample_data = data[i : i + window_t_r, :]
            # Correlate and convert to z
            corrs = np.corrcoef(sample_data.T)
            corrs = np.arctanh(corrs[idx])
            df = pd.DataFrame(columns=['fishers_z'], data=corrs)
            df['timepoint'] = i
            out_dfs.append(df)

        df = pd.concat(out_dfs)

        # Calculate mean + SD z-value by timepoint
        mean_by_timepoint = df.groupby(['timepoint'])['fishers_z'].mean()
        std_by_timepoint = df.groupby(['timepoint'])['fishers_z'].std()
        summary_df = pd.DataFrame(
            columns=['mean', 'std'],
            data=np.vstack((mean_by_timepoint.values, std_by_timepoint.values)).T,
        )

        # Save to file
        self._results['fc_inflation'] = os.path.abspath('fc_inflation.tsv')
        summary_df.to_csv(self._results['fc_inflation'], sep='\t', index=False)

        # Calculate percent increase in FC in the last window relative to the first window
        percent_fc_increase = 100 * mean_by_timepoint.iloc[-1] / mean_by_timepoint.iloc[0]

        # Calculate average FC in 2nd half minus avg FC in 1st half
        z_first_half = mean_by_timepoint.loc[: (mean_by_timepoint.shape[0] // 2) - 1].mean()
        z_second_half = mean_by_timepoint.loc[mean_by_timepoint.shape[0] // 2 :].mean()
        fc_diff = z_second_half - z_first_half

        metrics = {
            # percent increase in FC in the last window relative to the first window
            'percent_fc_increase': percent_fc_increase,
            # 'Avg FC in 2nd Half of Scan' minus 'Avg FC in 1st Half of Scan'
            'fc_diff': fc_diff,
            # correlation of FC with time
            # XXX: Should I average within timepoints first? Def changes the results.
            'fc_time_corr': df['fishers_z'].corr(df['timepoint']),
        }
        self._results['metrics'] = metrics

        return runtime
