"""Descriptive age-group classification using static and dynamic NII.

Inputs: Data_FC.csv, dynamic_NII_SALD.csv, dynamic_NII_LEMON.csv.
Rows must correspond to identical subjects in identical order within each cohort.
Features are precomputed; this script does not fit ICA or compute connectivity.
Within-cohort permutation tests retain the fixed-C procedure of the source script.
Cross-cohort evaluation fits scaling and tunes C using the source cohort only.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


FEAT = {'static': ['aDN_s', 'pDN_s'],
        'dynamic': ['aDN_dm', 'pDN_dm', 'aDN_dv', 'pDN_dv'],
        'combined': ['aDN_s', 'pDN_s', 'aDN_dm', 'pDN_dm', 'aDN_dv', 'pDN_dv']}
Cgrid = {'clf__C': np.logspace(-3, 3, 13)}


def build(st, cohort, dy):
    s = st[st.data == cohort].reset_index(drop=True)
    dy = dy.reset_index(drop=True)
    if len(s) != len(dy):
        raise ValueError(f'{cohort}: static and dynamic row counts differ')
    if not s['Group'].isin(['Young', 'Old']).all():
        raise ValueError(f'{cohort}: unexpected group labels')
    return pd.DataFrame({
        'group': s['Group'],
        'aDN_s': s['SN_CEN'] - s['SN_aDN'],
        'pDN_s': s['SN_CEN'] - s['SN_pvDN'],
        'aDN_dm': dy['NII_aDN_mean'], 'pDN_dm': dy['NII_pDN_mean'],
        'aDN_dv': dy['NII_aDN_std'], 'pDN_dv': dy['NII_pDN_std'],
    })


def nested_cv(X, y, seed):
    outer = StratifiedKFold(5, shuffle=True, random_state=seed)
    yp = np.zeros(len(y)); yprob = np.zeros(len(y))
    for tr, te in outer.split(X, y):
        inner = StratifiedKFold(5, shuffle=True, random_state=seed)
        pipe = Pipeline([('sc', StandardScaler()),
                         ('clf', LogisticRegression(class_weight='balanced', max_iter=2000))])
        gs = GridSearchCV(pipe, Cgrid, cv=inner, scoring='roc_auc')
        gs.fit(X[tr], y[tr])
        yp[te] = gs.predict(X[te]); yprob[te] = gs.predict_proba(X[te])[:, 1]
    return balanced_accuracy_score(y, yp), roc_auc_score(y, yprob)


def best_C(X, y):
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    pipe = Pipeline([('sc', StandardScaler()),
                     ('clf', LogisticRegression(class_weight='balanced', max_iter=2000))])
    return GridSearchCV(pipe, Cgrid, cv=cv, scoring='roc_auc').fit(X, y).best_params_['clf__C']


def fixedC_auc(X, y, C, seed=0):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    pipe = Pipeline([('sc', StandardScaler()),
                     ('clf', LogisticRegression(C=C, class_weight='balanced', max_iter=2000))])
    return roc_auc_score(y, cross_val_predict(pipe, X, y, cv=cv, method='predict_proba')[:, 1])


def permute_p(X, y, C, n_perm=1000, seed=123):
    obs = fixedC_auc(X, y, C)
    rng = np.random.RandomState(seed); cnt = 1
    for _ in range(n_perm):
        if fixedC_auc(X, rng.permutation(y), C) >= obs:
            cnt += 1
    return cnt / (n_perm + 1)

def cross_cohort(X_train, y_train, X_test, y_test):
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    pipe = Pipeline([('sc', StandardScaler()),
                     ('clf', LogisticRegression(class_weight='balanced', max_iter=2000))])
    gs = GridSearchCV(pipe, Cgrid, cv=cv, scoring='roc_auc')
    gs.fit(X_train, y_train)
    return (balanced_accuracy_score(y_test, gs.predict(X_test)),
            roc_auc_score(y_test, gs.predict_proba(X_test)[:, 1]))


def main():
    st = pd.read_csv('Data_FC.csv')
    cohorts = {c: build(st, c, pd.read_csv(f'dynamic_NII_{c}.csv'))
               for c in ('SALD', 'LEMON')}
    rows = []
    for source, df in cohorts.items():
        y = (df.group.values == 'Old').astype(int)
        for fs, cols in FEAT.items():
            X = df[cols].values
            accs, aucs = zip(*[nested_cv(X, y, seed) for seed in range(5)])
            p = permute_p(X, y, best_C(X, y), n_perm=1000)
            rows.append(dict(analysis='within', train=source, test=source,
                             feature_set=fs, bACC=np.mean(accs), AUC=np.mean(aucs),
                             perm_p=p, n_train_cohort=len(y), n_test_cohort=len(y)))
            for target, test in cohorts.items():
                if target == source:
                    continue
                yt = (test.group.values == 'Old').astype(int)
                acc, auc = cross_cohort(X, y, test[cols].values, yt)
                rows.append(dict(analysis='cross', train=source, test=target,
                                 feature_set=fs, bACC=acc, AUC=auc,
                                 perm_p=np.nan, n_train_cohort=len(y), n_test_cohort=len(yt)))
    results = pd.DataFrame(rows)
    print(results.to_string(index=False))
    results.to_csv('age_group_classification_results.csv', index=False)


if __name__ == '__main__':
    main()
