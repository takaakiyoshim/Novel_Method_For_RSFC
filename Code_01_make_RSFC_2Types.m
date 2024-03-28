function Code_01_make_RSFC_2Types(subNo,runNo,PWD)

%% generate 2 types RSFC data : traditional and novel methods

%% input %% 

%% WholeBrainData
%% Path : strcat( PWD, '/ResultData/Preprocessed_fmriData/100206/WholeBrainData_RunNo', num2str(runNo), '.mat' )
%% 2D Matrix 
%% fMRI data which has been preprocessed 
%% preprocessing including global signal regression
%% Row : 59412 vertives based on Glasser Atlas
%% Column : 1,100 time points

%% sublist
%% subject list in 995 subjects : For example 100206

%% ROI 
%% Path : strcat( PWD, '/Material/ROI.mat' )
%% cell size 360 x 1
%% each cell containg vertex number in 59412 vertices for each ROI 

%% output %% 

%% Corr_Matrix
%% 360 x 360 FC Matrices in both traditinal and novel methods

subNo = str2num(subNo);
runNo = str2num(runNo);

%% load some materials 
sublist = dlmread( strcat( PWD, '/Material/SubjectIdList995SubjectsAllRunsCompleted.txt' ) );
load( strcat( PWD, '/Material/ROI.mat' ) );
%% ridge parameter was set 0.1
BestParam = 0.1;

%% set path
InputPath = strcat (  PWD, '/Preprocessed_fmriData/', num2str(sublist(subNo)) );
OutputPath_Traditional = strcat( PWD, '/ResultData/FC_Matrix_Traditional_Method/', num2str(sublist(subNo)) );
OutputPath_Novel = strcat( PWD, '/ResultData/FC_Matrix_Novel_Method/', num2str(sublist(subNo)) );
mkdir ( OutputPath_Traditional );
mkdir ( OutputPath_Novel );

%% load data    
load ( strcat( InputPath, '/WholeBrainData_RunNo', num2str(runNo), '.mat' ) );

%% zscore across runs
RawData = zscore(WholeBrainData,0,2);
clear WholeBrainData;

%% old method : Sliding Window analysis 
DataAveragedWithinROI = NaN(360,size(RawData,2));

for s = 1:360
    data = RawData(ROI{s},:);
    DataAveragedWithinROI(s,:) = mean(data,1);
end
clear data s;

%% set output variable
Corr_Matrix = NaN(360,360,[size(RawData,2)-32+1]);

for s = 1:(size(RawData,2)-32+1)
    data = DataAveragedWithinROI(:,[s:(s+32-1)]);
    Corr_Matrix(:,:,s) = corr(data');
end
clear data s;

%% save data in average 
Corr_Matrix = mean(Corr_Matrix,3);
save( strcat( OutputPath_Traditional, '/Corr_Matrix_RunNo', num2str(runNo), '.mat' ), 'Corr_Matrix' );
clear Corr_Matrix;

%% novel method
%% set ouput variable
Corr_Matrix = NaN(360,360,5);

%% 5 fold cross validation
for iterCV = 1:5
    
    %% set target data
    TargetTrainingData = RawData;
    TargetTrainingData(:,[(1+(iterCV-1)*(1100/5)):(iterCV*(1100/5))]) = [];
    TargetTrainingData = TargetTrainingData';

    TargetTestData = RawData(:,[(1+(iterCV-1)*(1100/5)):(iterCV*(1100/5))]);

    for iterSeed = 1:360

       %% set seed data
       SeedTrainingData = RawData(ROI{iterSeed},:);
       SeedTrainingData(:,[(1+(iterCV-1)*(1100/5)):(iterCV*(1100/5))]) = [];
       SeedTrainingData = SeedTrainingData';      

       SeedTestData = RawData(ROI{iterSeed},[(1+(iterCV-1)*(1100/5)):(iterCV*(1100/5))]);

       %% ridge regression
       %% we avoid to use matlab's function "ridge", because we can process multple regression calculations simultaneouly by using this function
       betaSet = pinv(SeedTrainingData'*SeedTrainingData+BestParam.*eye(size(SeedTrainingData,2)))*SeedTrainingData'*TargetTrainingData; 
       clear  SeedTrainingData;

       PredictedData = betaSet'*SeedTestData;
       clear SeedTestData;

       %% spatial pattern
         for iterTarget = 1:360
             if ( isequal(iterSeed,iterTarget) == 0 )
                Data = NaN(1,(1100/5));
               for t = 1:(1100/5)
                   %% we avoid to use matlab's function "corr" or "corrcoef", because of their slow processing
                   DemeanPredicted = [PredictedData(ROI{iterTarget},t) - mean(PredictedData(ROI{iterTarget},t))];
                   DemeanData = [TargetTestData(ROI{iterTarget},t) - mean(TargetTestData(ROI{iterTarget},t))];    
                   Data(t) = sum(DemeanPredicted.*DemeanData)/(rssq(DemeanPredicted)*rssq(DemeanData));
               end
               Corr_Matrix(iterSeed,iterTarget,iterCV) = mean(Data);
            end
         end
         clear Data; clear DemeanPredicted; clear DemeanData;

    end
    
end
%% save data in average 
Corr_Matrix = mean(Corr_Matrix,3);
save( strcat( OutputPath_Novel, '/Corr_Matrix_RunNo', num2str(runNo), '.mat' ), 'Corr_Matrix' );
      