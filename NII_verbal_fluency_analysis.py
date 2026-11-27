"""Pearson correlations between NII temporal variability and verbal fluency.

LEMON cohort only. Requires pandas, numpy and scipy.
Input files must contain the same participants in the same row order.
Edit the behavioral filename and column names to match the source data.
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, norm

DYN_LEMON = 'dynamic_NII_decomposed_LEMON_corrected.csv'
FLUENCY_LEMON = 'verbal_fluency_LEMON.csv'
FLUENCY = ['phonemic_fluency', 'semantic_fluency']
NII = ['NII_std', 'NII_aDN_std', 'NII_pDN_std']


def correlation(df, x, y):
    """Two-sided Pearson correlation using complete pairs."""
    pairs = df[[x, y]].dropna()
    n = len(pairs)
    if n < 3 or pairs[x].nunique() < 2 or pairs[y].nunique() < 2:
        return n, np.nan, np.nan
    r, p = pearsonr(pairs[x], pairs[y])
    return n, r, p


def compare_correlations(n_y, r_y, n_o, r_o):
    """Fisher z test for correlations from independent age groups."""
    if min(n_y, n_o) <= 3 or not (abs(r_y) < 1 and abs(r_o) < 1):
        return np.nan, np.nan
    z = (np.arctanh(r_y) - np.arctanh(r_o)) / np.sqrt(
        1 / (n_y - 3) + 1 / (n_o - 3))
    return z, 2 * norm.sf(abs(z))


def main():
    nii = pd.read_csv(DYN_LEMON).rename(columns={'grp': 'Group'})
    fluency = pd.read_csv(FLUENCY_LEMON)
    if len(nii) != len(fluency):
        raise ValueError('Input files have different numbers of rows.')
    if not nii['Group'].isin(['Young', 'Old']).all():
        raise ValueError('Expected Group labels: Young and Old.')
    # Row alignment must be verified against participant IDs before running.
    df = pd.concat([nii[['Group'] + NII].reset_index(drop=True),
                    fluency[FLUENCY].reset_index(drop=True)], axis=1)
    groups = {'Young': df[df.Group == 'Young'],
              'Old': df[df.Group == 'Old'], 'Combined': df}
    correlations, comparisons = [], []
    for x in NII:
        for y in FLUENCY:
            results = {}
            for group, subset in groups.items():
                n, r, p = correlation(subset, x, y)
                results[group] = (n, r, p)
                correlations.append(dict(NII=x, fluency=y, group=group,
                                         n=n, r=r, p=p))
                print(f'{x} | {y} | {group}: n={n}, r={r:.4f}, p={p:.4g}')
            ny, ry, _ = results['Young']
            no, ro, _ = results['Old']
            z, p = compare_correlations(ny, ry, no, ro)
            comparisons.append(dict(NII=x, fluency=y, n_young=ny,
                                    n_old=no, r_young=ry, r_old=ro, z=z, p=p))
            print(f'  Young vs Old: Z={z:.4f}, p={p:.4g}')
    pd.DataFrame(correlations).to_csv('NII_fluency_correlations.csv', index=False)
    pd.DataFrame(comparisons).to_csv('NII_fluency_Fisher_z.csv', index=False)


if __name__ == '__main__':
    main()
