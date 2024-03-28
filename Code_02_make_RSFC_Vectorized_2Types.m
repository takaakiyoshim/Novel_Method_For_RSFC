function Code_02_make_RSFC_Vectorized_2Types(PWD)

%% step1  convert 360 x 360 FC matrices in 995 x 4 samples to 2D matrix
%% dimension of 2D is 359 x 180 in traditional FC due to the matrix' symmetry
%% dimension of 2D is 359 x 360 in novel FC due to the matrix' asymmetry

%% step2 Sex and age effects were regressed out from these data

%% Input %% 
%% FC Matrix in both traditional and novel methods
%% unrestricted_file.csv : subject's data file including age and sex provided by HCP

%% load subject label and  gender, age variable
sublist = dlmread( strcat( PWD, '/Material/SubjectIdList995SubjectsAllRunsCompleted.txt' ) );
InputPath_Traditional = strcat( PWD, '/ResultData/FC_Matrix_Traditional_Method/', num2str(sublist(subNo)) );
InputPath_Novel = strcat( PWD, '/ResultData/FC_Matrix_Novel_Method/', num2str(sublist(subNo)) );
OutputPath = strcat( PWD, '/ResultData/Vectorized_FC' );


tab = readtable('/mnt/nasNew8_Yoshimoto/NovelMethodRidge02/Materials/unrestricted_file.csv');
Sub = table2cell(tab(:,1));
Sub = cell2mat(Sub);
Sex = table2cell(tab(:,4));Data
Age = table2cell(tab(:,5));

SexVec = NaN(995,1);
AgeVec = NaN(995,1);

UniqueAge = unique(Age);

for s = 1:995
    
    Numb = find(Sub==SubjectIdList(s));
    
    if ( Sex{Numb} == 'M' )
       SexVec(s,1) = 1;
    elseif ( Sex{Numb} == 'F' )
        SexVec(s,1) = 2;
    end
    
    for t = 1:length(UniqueAge)
        if ( strcmp(Age{Numb},UniqueAge{t}) == 1 )
            AgeVec(s,1) = t;
        end
    end
end

SexVecCopy = NaN(995*4,1);
AgeVecCopy = NaN(995*4,1);

for s = 1:995
    SexVecCopy([(1+(s-1)*4):s*4],1) = repmat(SexVec(s,1),[4,1]);
    AgeVecCopy([(1+(s-1)*4):s*4],1) = repmat(AgeVec(s,1),[4,1]);
end

SexVec = SexVecCopy; clear SexVecCopy;
AgeVec = AgeVecCopy; clear AgeVecCopy;
    
%% traditional method %%

%% step 1
FC = NaN(180*359,995*4);

for s = 1:995
    for t = 1:4
        load( strcat( InputPath_Traditional,'/', num2str(sublist(s)), '/Corr_Matrix_RunNo', num2str(t), '.mat' ), 'Corr_Matrix' );
    
        %% extract elements in upper triangle
        TempCorr = triu(CorrMatrix,1); clear Corr_Matrix;
        TempCorr(TempCorr==0) = NaN;
        TempCorr = reshape(TempCorr,[],1);
        TempCorr = TempCorr(~isnan(TempCorr));
        
        FC(:,(4*(s-1)+t)) = TempCorr;  clear TempCorr;    
    end
end

save( strcat( OutputPath, '/FC_Traditional.mat' ), 'FC', '-v7.3' );

%% step 2
FC_Copy = NaN(size(FC,1),size(FC,2));

for t = 1:size(FC,1)
    [b,dev,stats] = glmfit(cat(2,SexVec,AgeVec),FC(t,:)');
    FC_Copy(t,:) = stats.resid';
end
clear b dev stats;

FC = FC_Copy; clear FC_Copy;
save( strcat( OutputPath, '/FC_Traditional_GenderAgeRegressedOut.mat' ), 'FC', '-v7.3' );
clear FC;

%% novel method %%

FC = NaN(360*359,995*4);

for s = 1:995
    for t = 1:4
        load( strcat( InputPath_Novel,'/', num2str(sublist(s)), '/Corr_Matrix_RunNo', num2str(t), '.mat' ), 'Corr_Matrix' );
    
        %% all diagonal elements are NaNs 
        TempCorr = reshape(CorrMatrix,,[],1); clear Corr_Matrix;
        TempCorr = TempCorr(~isnan(TempCorr));
        
        FC(:,(4*(s-1)+t)) = TempCorr;  clear TempCorr;    
    end
end

save( strcat( OutputPath, '/FC_Novel.mat' ), 'FC', '-v7.3' );

%% step 2
FC_Copy = NaN(size(FC,1),size(FC,2));

for t = 1:size(FC,1)
    [b,dev,stats] = glmfit(cat(2,SexVec,AgeVec),FC(t,:)');
    FC_Copy(t,:) = stats.resid';
end
clear b dev stats;


FC = FC_Copy; clear FC_Copy;
save( strcat( OutputPath, '/FC_Novel_GenderAgeRegressedOut.mat' ), 'FC', '-v7.3' );
clear FC;