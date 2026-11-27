%% Dynamic network interaction index (NII): LEMON cohort
% Calculate dynamic NII from precomputed GIFT dFNC results.
% Requires MATLAB and the Statistics and Machine Learning Toolbox.
%
% Inputs:
%   Lemon_Rest_dfnc_sub_%03d_sess_001_results.mat (subjects 001-133)
%   FNCdyn: windows x 10 matrix of correlations (not Fisher z-transformed).
%   age.mat: age labels (1 = younger, 2 = older), in subject file order.
%
% Network order: 1 = lCEN, 2 = rCEN, 3 = SN, 4 = pDN, 5 = aDN.
% GIFT lower-triangle vectorization (column-major):
%   SN-lCEN = column 2; SN-rCEN = column 5;
%   SN-pDN = column 8; SN-aDN = column 9.
%
% Fisher r-to-z transformation is applied before calculating NII.
% NII = SN_CEN - SN_DMN; subsystem indices use SN_aDN or SN_pDN.
% Dynamic mean and temporal variability are the mean and sample SD
% across windows, respectively.
% Group comparisons: Mann-Whitney U test (approximate method) and
% Cohen's d (younger minus older, divided by pooled SD).
% Reported p-values are uncorrected.

%% Input configuration
datadir = '.';                 % folder containing the 133 results files
N = 133;
load('age.mat');               % age: N x 1 (1=younger, 2=older), aligned sub_001..133

%% Calculate subject-level dynamic NII
z = @atanh;
NIIm=zeros(N,1); NIIs=zeros(N,1);
pDNm=zeros(N,1); pDNs=zeros(N,1);
aDNm=zeros(N,1); aDNs=zeros(N,1);

for s = 1:N
    fn = fullfile(datadir, sprintf('Lemon_Rest_dfnc_sub_%03d_sess_001_results.mat', s));
    R  = load(fn);             % loads FNCdyn etc.
    F  = R.FNCdyn;             % windows x 10
    SN_CEN = (z(F(:,2)) + z(F(:,5)))/2;
    SN_pDN = z(F(:,8));
    SN_aDN = z(F(:,9));
    SN_DMN = (SN_pDN + SN_aDN)/2;
    NII    = SN_CEN - SN_DMN;
    NII_pDN= SN_CEN - SN_pDN;
    NII_aDN= SN_CEN - SN_aDN;
    NIIm(s)=mean(NII);    NIIs(s)=std(NII);
    pDNm(s)=mean(NII_pDN);pDNs(s)=std(NII_pDN);
    aDNm(s)=mean(NII_aDN);aDNs(s)=std(NII_aDN);
end

%% Group-level statistical comparisons
young=(age(:)==1); old=(age(:)==2);
meas = {'NII dyn-mean',NIIm; 'NII_pDN dyn-mean',pDNm; 'NII_aDN dyn-mean',aDNm; ...
        'NII dyn-var', NIIs; 'NII_pDN dyn-var', pDNs; 'NII_aDN dyn-var', aDNs};
fprintf('N=%d (younger=%d, older=%d); Fisher z\n\n', N, sum(young), sum(old));
fprintf('%-18s %16s %16s %7s %9s %7s\n','measure','young(M +/- SD)','old(M +/- SD)','z','p','d');
for k=1:size(meas,1)
    v=meas{k,2}; y=v(young); o=v(old);
    [p,~,st]=ranksum(y,o,'method','approximate');
    sp=sqrt(((numel(y)-1)*var(y)+(numel(o)-1)*var(o))/(numel(y)+numel(o)-2));
    d=(mean(y)-mean(o))/sp;
    fprintf('%-18s %+8.3f +/- %.3f %+8.3f +/- %.3f %+7.2f %9.4f %+7.2f\n', ...
        meas{k,1},mean(y),std(y),mean(o),std(o),st.zval,p,d);
end