function Code_03_predict_Trait_Behavior_By_SVM(method_name,Behavior_No,PWD)

%% argument %% 

%% method_name "Traditional" or "Novel"

%% Behavior_No integer from 1 To 60
%% Behavior_No from 1 to 58 represent behavioral measures in HCP data
%% Behavior_No 59 sex difference
%% Behavior_No 60 age

Behavior_No = str2num(Behavior_No);

%% load random number
load( strcat( PWD, '/Material/CVpartition.mat' ) );

%% load subject list
sublist = dlmread( strcat( PWD, '/Material/SubjectIdList995SubjectsAllRunsCompleted.txt' ) );

%% load dependent variable
if ( Behavior_No < 59 )
    load( strcat( PWD, '/Material/Data_58BehavioralMeasurements_995Subjects_GenderAgeRegressedout_Double.mat' ) );
    Y = Data_58BehavioralMeasurements(:,Behavior_No); clear Data_58BehavioralMeasurements;

    %% mulitply data and for 4 sessions
    Y_Multiplied = NaN(size(Y,1)*4,1);

    for s = 1:size(Y,1)
        Y_Multiplied([(1+4*(s-1)):(4*s)],1) = repmat(Y(s,1),[4,1]);
    end
    clear s;
elseif ( Behavior_No == 59 )
    load( strcat( PWD, '/Material/SubjectGenderList995SubjectsAllRunsCompleted.mat' ) );
    
    %% mulitply data and for 4 sessions
     Y_Multiplied = cell(1,995*4);
    for s = 1:995
        for t = 1:4
           Y_Multiplied{1,(t+4*(s-1))} = SubjectGenderList{s};
        end
    end
    clear s; clear t;
    
elseif ( Behavior_No == 60 )
    load( strcat( PWD, '/Material/AgeVector995SubjectsAllRunsCompleted.mat' ) );
    Y = AgeVector; clear AgeVector;
    
    %% mulitply data and for 4 sessions
    Y_Multiplied = NaN(size(Y,1)*4,1);

    for s = 1:size(Y,1)%% Behavior_No
        Y_Multiplied([(1+4*(s-1)):(4*s)],1) = repmat(Y(s,1),[4,1]);
    end
    clear s;
else
    return;
end

%% load independent variable : RSFC
if ( Behavior_No < 59 )
   load( strcat( PWD, '/ResultData/Vectorized_FC/FC_', method_name, 'GenderAgeRegressedOut.mat' ) );
else
   load( strcat( PWD, '/ResultData/Vectorized_FC/FC_', method_name, '.mat' ) );
end


%% set output path
OutputPath = strcat( PWD, '/ResultData/Predicted_Score_Single_Method/', method_name, '_Method' );
    
%% make result container
MeanPredictedScores = NaN(995,1);
PredictedScores = NaN(995*4,1);

    
%% zscore feature direction
X = zscore(FC,0,1); clear FC;

if ( Behavior_No ~= 59 )
    
    %% solve regression problem
    %% 10 fold cross validation 
    %% using fitrsvm

    for iterCV = 1:10
      
        %% set cv %%
        TrainCV = CVpartion{iterCV,1};
        TestCV = CVpartion{iterCV,2};

        TrainCV_Multiplied = [];
        TestCV_Multiplied = [];

        for s = 1:length(TrainCV)
            TrainCV_Multiplied = cat(1,TrainCV_Multiplied,[(4*TrainCV(s,1)-3):4*TrainCV(s,1)]');
        end

        for s = 1:length(TestCV)
            TestCV_Multiplied = cat(1,TestCV_Multiplied,[(4*TestCV(s,1)-3):4*TestCV(s,1)]');
        end
        clear s;
        %%%%%%%%%%%%

        %% divide X, Y
        X_Training = X(:,TrainCV_Multiplied);
        X_Test = X(:,TestCV_Multiplied);

        Y_Training = Y_Multiplied(TrainCV_Multiplied);
 
        %% model fitting
        Mdl = fitrsvm(X_Training',Y_Training);
              
        %% prediction
        Score = predict(Mdl,X_Test');

        PredictedScores(TestCV_Multiplied) = Score;
       
    end

    for iterSub = 1:995
        MeanPredictedScores(iterSub,1) = mean(PredictedScores([(1+4*(iterSub-1)):4*iterSub]));
    end
    
    if ( Behavior_No ~= 60 )
      save( strcat( OutputPath, '/Result_Behavior_No', num2str(Behavior_No), '.mat'), 'PredictedScores','MeanPredictedScores');
    else
      save( strcat( OutputPath, '/Result_Age.mat'), 'PredictedScores','MeanPredictedScores'); 
    end
    
else  
    %% solve classification problem for sex difference
    %% 10 fold cross validation 
    %% using fitcsvm function

    for iterCV = 1:10

        %% set cv %%
        TrainCV = CVpartion{iterCV,1};
        TestCV = CVpartion{iterCV,2};

        TrainCV_Multiplied = [];
        TestCV_Multiplied = [];

        for s = 1:length(TrainCV)
            TrainCV_Multiplied = cat(1,TrainCV_Multiplied,[(4*TrainCV(s,1)-3):4*TrainCV(s,1)]');
        end

        for s = 1:length(TestCV)
            TestCV_Multiplied = cat(1,TestCV_Multiplied,[(4*TestCV(s,1)-3):4*TestCV(s,1)]');
        end
        clear s;
        %%%%%%%%%%%%

        %% divide X, Y
        X_Training = X(:,TrainCV_Multiplied);
        X_Test = X(:,TestCV_Multiplied);

        Y_Training = cell(1,length(TrainCV_Multiplied));

        for s = 1:length(TrainCV_Multiplied)
           Y_Training{1,s} = Y_Multiplied{1,TrainCV_Multiplied(s)};
        end
        clear s;
 
        %% model fitting
        Mdl = fitcsvm(X_Training',Y_Training);
    
        %% estimation
        [~,Score] = predict(Mdl,X_Test');

        PredictedScores(TestCV_Multiplied) = Score(:,1);

    end

    for s = 1:995
        MeanPredictedScores(s,1) = mean(PredictedScores([(1+4*(s-1)):4*s]));
    end
    clear s;

    PredictedLabel = cell(1,995);

    for s = 1:995
        if ( MeanPredictedScores(s) >= 0 )
            PredictedLabel{s} = 'F';
        else
            PredictedLabel{s} = 'M'; 
        end
    end
    clear s;

    C = confusionmat(SubjectGenderList,PredictedLabel);

    PredictiveAccuracy = (C(1,1) + C(2,2)) / sum(sum(C));

    save( strcat( OutputPath, '/Result_Sex.mat'),...
        'PredictedScores','MeanPredictedScores','PredictedLabel','PredictiveAccuracy');

end    

    
    
