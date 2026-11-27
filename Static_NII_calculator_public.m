%% Static network interaction index (NII)
% LEMON dataset: joint ICA decomposition.
% Inputs in the MATLAB workspace:
%   fncYoung: 88 x 10 matrix of raw Pearson correlations.
%   fncOld:   45 x 10 matrix of raw Pearson correlations.
% Do not use inputs that have already been Fisher z-transformed.
% Network order: 1 = lCEN, 2 = rCEN, 3 = SN, 4 = pDN, 5 = aDN.
% Edge order: (1,2), (1,3), (1,4), (1,5), (2,3), (2,4),
%             (2,5), (3,4), (3,5), (4,5).
% Confirm the edge order against the original connectivity data.
% Requires MATLAB Statistics and Machine Learning Toolbox.

%% Fisher r-to-z transformation
fncYoung_z = atanh(fncYoung);
fncOld_z = atanh(fncOld);

%% Edge indices
iSNlCEN = 2;
iSNrCEN = 5;
iSNpDN = 8;
iSNaDN = 9;

%% Calculate static NII and subsystem-specific indices
calc = @(F) deal( ...
    (F(:,iSNlCEN) + F(:,iSNrCEN))/2, ...
    F(:,iSNpDN), ...
    F(:,iSNaDN));

[SNCEN_y, SNpDN_y, SNaDN_y] = calc(fncYoung_z);
[SNCEN_o, SNpDN_o, SNaDN_o] = calc(fncOld_z);

SNDMN_y = (SNpDN_y + SNaDN_y)/2;
SNDMN_o = (SNpDN_o + SNaDN_o)/2;

NII_y = SNCEN_y - SNDMN_y;
NII_o = SNCEN_o - SNDMN_o;
NIIpDN_y = SNCEN_y - SNpDN_y;
NIIpDN_o = SNCEN_o - SNpDN_o;
NIIaDN_y = SNCEN_y - SNaDN_y;
NIIaDN_o = SNCEN_o - SNaDN_o;

%% Group comparisons
% Two-sided Mann-Whitney U test (Wilcoxon rank-sum test).
% Cohen's d = (younger - older) / pooled standard deviation.
% Negative d indicates higher values in older adults.
% Reported p-values are uncorrected.
names = {'NII','NII_pDN','NII_aDN','SN_CEN','SN_pDN','SN_aDN'};
Y = {NII_y,NIIpDN_y,NIIaDN_y,SNCEN_y,SNpDN_y,SNaDN_y};
O = {NII_o,NIIpDN_o,NIIaDN_o,SNCEN_o,SNpDN_o,SNaDN_o};

fprintf('\n%-12s %18s %18s %8s %10s %8s\n', ...
    'Measure','Young (M +/- SD)','Old (M +/- SD)','z','p','d');
for k = 1:numel(names)
    y = Y{k};
    o = O{k};
    [p,~,st] = ranksum(y,o,'method','approximate');
    sp = sqrt(((numel(y)-1)*var(y) + (numel(o)-1)*var(o)) / ...
        (numel(y)+numel(o)-2));
    d = (mean(y)-mean(o))/sp;
    fprintf('%-12s %+8.3f +/- %.3f %+8.3f +/- %.3f %+8.2f %10.4f %+8.2f\n', ...
        names{k},mean(y),std(y),mean(o),std(o),st.zval,p,d);
end
