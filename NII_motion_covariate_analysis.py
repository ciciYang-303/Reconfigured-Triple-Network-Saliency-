# =====================================================================
#  Does the age effect on the NII survive adjustment for head motion + sex?
#  Models:  NII ~ group           (raw)
#           NII ~ group + FD + sex (adjusted)
#  Edit the FILE PATHS, then run.  Requires: pandas, numpy, statsmodels, scipy
# =====================================================================
import pandas as pd, numpy as np
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu, chi2_contingency

STATIC_FC = 'Data_FC.csv'                                  # cols: data, Group, SN_CEN, SN_pvDN, SN_aDN, diff_NII ...
DYN_SALD  = 'dynamic_NII_decomposed_SALD_corrected.csv'    # cols: grp, NII_mean, NII_pDN_mean ...
DYN_LEMON = 'dynamic_NII_decomposed_LEMON_corrected.csv'
FD_SALD   = 'FD_SALD.csv'                                  # cols: group, Sex, FD_SALD   (subject order == NII order!)
FD_LEMON  = 'FD_LEMON.csv'                                 # cols: group, gender, FD

def load_fd(path, fdcol, sexcol):
    fd = pd.read_csv(path).dropna(how='all')
    fd = fd[fd['group'].notna() & fd[fdcol].notna()].reset_index(drop=True)
    return fd.rename(columns={fdcol:'FD', sexcol:'sex'})[['group','sex','FD']]

st = pd.read_csv(STATIC_FC); st['NII_pDN'] = st['SN_CEN'] - st['SN_pvDN']
fdS = load_fd(FD_SALD, 'FD_SALD', 'Sex')
fdL = load_fd(FD_LEMON, 'FD', 'gender')

def cohend(a,b):
    a,b=np.asarray(a),np.asarray(b)
    sp=np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
    return (a.mean()-b.mean())/sp
def pf(p): return '<.001' if p<.001 else f'{p:.3f}'

for coh, fd, dynf in [('SALD', fdS, DYN_SALD), ('LEMON', fdL, DYN_LEMON)]:
    nii = pd.read_csv(dynf).rename(columns={'grp':'Group'})
    s   = st[st['data']==coh].reset_index(drop=True)
    assert len(fd)==len(nii)==len(s), f'{coh}: row mismatch {len(fd)},{len(nii)},{len(s)}'
    dyn  = pd.concat([fd.reset_index(drop=True), nii.reset_index(drop=True)], axis=1)
    stat = pd.concat([fd.reset_index(drop=True), s[['Group','diff_NII','NII_pDN']].reset_index(drop=True)], axis=1)
    print(f'\n===== {coh} =====')
    Y=fd[fd.group==1]['FD']; O=fd[fd.group==2]['FD']; _,pfd=mannwhitneyu(Y,O)
    ct=pd.crosstab(fd.group,fd.sex); chi2,pc,_,_=chi2_contingency(ct)
    print(f'  FD: Young={Y.mean():.3f}, Old={O.mean():.3f}, d={cohend(Y,O):+.2f}, p={pf(pfd)} | sex chi2={chi2:.2f}, p={pf(pc)}')
    for meas, df, dv in [('NII static',stat,'diff_NII'),('NII_pDN static',stat,'NII_pDN'),
                         ('NII dynamic',dyn,'NII_mean'),('NII_pDN dynamic',dyn,'NII_pDN_mean')]:
        d=df.copy(); d['grp']=(d['Group']=='Old').astype(int); d['FDz']=(d.FD-d.FD.mean())/d.FD.std()
        m0=smf.ols(f'{dv} ~ grp', d).fit()
        m1=smf.ols(f'{dv} ~ grp + FDz + C(sex)', d).fit()
        psex=[v for k,v in m1.pvalues.items() if 'sex' in k][0]
        print(f'  {meas:16s}: b_grp raw={m0.params["grp"]:+.3f} (p={pf(m0.pvalues["grp"])}) | '
              f'+FD+sex b_grp={m1.params["grp"]:+.3f} (p={pf(m1.pvalues["grp"])}) | '
              f'p_FD={pf(m1.pvalues["FDz"])}, p_sex={pf(psex)}')
