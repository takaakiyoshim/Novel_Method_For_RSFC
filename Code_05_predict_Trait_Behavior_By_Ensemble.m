function Code_05_predict_Trait_Behavior_By_Ensemble(Behavior_No,PWD)


Behavior_No = str2num(Behavior_No);

%% load random number
load( strcat( PWD, '/Material/CVpartition.mat' ) );

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

%% acquire input path
X_Traditional_Outer_CV_Path = strcat( PWD, '/ResultData/Predicted_Score_Single_Method/Traditional_Method' );
X_Novel_Outer_CV_Path = strcat( PWD, '/ResultData/Predicted_Score_Single_Method/Novel_Method' );

if ( Behavior_No < 59 )
   
   X_Traditional_Nested_CV_Path = strcat( PWD, '/ResultData/Predicted_Score_Nested_CV_For_Ensemble/Traditional_Method/Predicted_Score_Behavior_No', num2str(Behavior_No) );
   X_Novel_Nested_CV_Path = strcat( PWD, '/ResultData/Predicted_Score_Nested_CV_For_Ensemble/Novel_Method/Predicted_Score_Behavior_No', num2str(Behavior_No) );
  
elseif ( Behavior_No == 59 ) 
       
   X_Traditional_Nested_CV_Path = strcat( PWD, '/ResultData/Predicted_Score_Nested_CV_For_Ensemble/Traditional_Method/Predicted_Score_Sex' );
   X_Novel_Nested_CV_Path = strcat( PWD, '/ResultData/Predicted_Score_Nested_CV_For_Ensemble/Novel_Method/Predicted_Score_Sex' );
  
   
elseif ( Behavior_No == 60 ) 
 
   X_Traditional_Nested_CV_Path = strcat( PWD, '/ResultData/Predicted_Score_Nested_CV_For_Ensemble/Traditional_Method/Predicted_Score_Age' );
   X_Novel_Nested_CV_Path = strcat( PWD, '/ResultData/Predicted_Score_Nested_CV_For_Ensemble/Novel_Method/Predicted_Score_Age' );
  
end   

%% load scores single method
if ( Behavior_No < 59 )
    load (strcat( X_Traditional_Outer_CV_Path, '/Result_Behavior_No', num2str(Behavior_No), '.mat' ) ); 
    X_Traditional_Outer_CV = PredictedScores;
    load (strcat( X_Novel_Outer_CV_Path, '/Result_Behavior_No', num2str(Behavior_No), '.mat' ) );
    X_Novel_Outer_CV = PredictedScores; clear PredictedScores; 
elseif ( Behavior_No == 59 )
    load (strcat( X_Traditional_Outer_CV_Path, '/Result_Sex.mat' ) ); 
    X_Traditional_Outer_CV = PredictedScores;
    load (strcat( X_Novel_Outer_CV_Path, '/Result_Sex.mat' ) ); 
    X_Novel_Outer_CV = PredictedScores; clear PredictedScores; 
elseif ( Behavior_No == 60 )
   load (strcat( X_Traditional_Outer_CV_Path, '/Result_Age.mat' ) ); 
    X_Traditional_Outer_CV = PredictedScores;
    load (strcat( X_Novel_Outer_CV_Path, '/Result_Age.mat' ) ); 
    X_Novel_Outer_CV = PredictedScores; clear PredictedScores; 
end

%% make result container
MeanPredictedScores = NaN(995,1);
PredictedScoresCopy = NaN(995*4,1);


%% 10 fold cross validation 

for iterCV = 1:10

    %% set cv %%
    TrainCV = CVpartition{iterCV,1};
    TestCV = CVpartition{iterCV,2};

    TrainCV_Multiplied = [];
    TestCV_Multiplied = [];

    for s = 1:length(TrainCV)
        TrainCV_Multiplied = cat(1,TrainCV_Multiplied,[(4*TrainCV(s,1)-3):4*TrainCV(s,1)]');
    end

    for s = 1:length(TestCV)
        TestCV_Multiplied = cat(1,TestCV_Multiplied,[(4*TestCV(s,1)-3):4*TestCV(s,1)]');
    end
    clear s;
    
    %% set output path
    
 
    %% load nested CV data
    load( strcat( PWD, '/ResultData/Predicted_Score_Nested_CV_For_Ensemble/Traditional_Method/PredictedScores_CV_No', num2str(iter_CV_No), '.mat' ), 'PredictedScores');
    X_Traditional_Nested_CV = NaN(995*4,1);
    X_Traditional_Nested_CV(TrainCV_Multiplied,1) = PredictedScores; clear PredictedScores;
   
    load( strcat( PWD, '/ResultData/Predicted_Score_Nested_CV_For_Ensemble/Novel_Method/PredictedScores_CV_No', num2str(iter_CV_No), '.mat' ), 'PredictedScores');
    X_Novel_Nested_CV = NaN(995*4,1);
    X_Novel_Nested_CV(TrainCV_Multiplied,1) = PredictedScores; clear PredictedScores;
   
    %% estimate 2 weights in 9 inner fold loop   
    BetaSet = zeros(2,1);
    
     for iterInCV = 1:10
            tic
            if ( isequal( iterCV, iterInCV ) == 0 )

                %% set inner cv
                InValidationCV = CVpartition{iterInCV,2};
                InValidationCV_Multiplied = [];
                for s = 1:length(InValidationCV)
                    InValidationCV_Multiplied = cat(1,InValidationCV_Multiplied,[(4*InValidationCV(s,1)-3):4*InValidationCV(s,1)]');
                end
                clear s;
                InTrainCV_Multiplied = TrainCV_Multiplied;

                for s = 1:length(InValidationCV_Multiplied)
                    InTrainCV_Multiplied(find(InTrainCV_Multiplied==InValidationCV_Multiplied(s))) = [];
                end
                clear s;

                %% divide X, Y
                X_Validation_Traditional = X_Traditional_Nested_CV(InValidationCV_Multiplied);
                X_Validation_Novel = X_Novel_Nested_CV(InValidationCV_Multiplied);
                Y_Validation = Y_Multiplied(InValidationCV_Multiplied);
                
                NaN_Loc = find(isnan(Y_Validation));
                Y_Validation(NaN_Loc) = [];
                X_Validation_Traditional(NaN_Loc) = [];
                X_Validation_Novel(NaN_Loc) = [];
                
                %% zscore
                X_Validation_Traditional = zscore(X_Validation_Traditional);
                X_Validation_Novel = zscore(X_Validation_Novel);
                if ( Behavior_No ~= 59 )
                  Y_Validation = zscore(Y_Validation);
                end

                %% constrained linear least-squares problem
                %% beta1, beta2 >= 0 and beta1 + beta2 = 1
                Beta = lsqlin(cat(2,X_Validation_Traditional,X_Validation_Novel),Y_Validation,[],[],ones(1,2),1,zeros(1,2),ones(1,2));
                BetaSet = BetaSet + Beta; 
                
            end
     end
        
    BetaSet = BetaSet ./ 9;
    BetaSet = BetaSet';
   
    Predicted_Score_Pair = cat(2,zscore(X_Traditional_Outer_CV(TestCV_Multiplied)),zscore(X_Novel_Outer_CV(TestCV_Multiplied)));
   
    BetaSet = repmat(BetaSet,[size( Predicted_Score_Pair,1),1]);
    
    %% calculate weighted average
    PredictedScoresCopy(TestCV_Multiplied) = sum((BetaSet.* Predicted_Score_Pair),2);
    
end
PredictedScores = PredictedScoresCopy; clear PredictedScoresCopy; 

for s = 1:995
    MeanPredictedScores(s,1) = mean(PredictedScores([(1+(s-1)*4):(s*4)],1));
end

%% save data
OutputPath = strcat( PWD, '/ResultData/Predicted_Score_Ensemble' );

if ( Behavior_No < 59 )
     
   save( strcat( OutputPath, '/Result_Behavior_No',num2str(Behavior_No),'.mat'), 'PredictedScores', 'MeanPredictedScores' );
   
elseif ( Behavior_No == 59 )
    
   %% load subject gender list 
   load( strcat( PWD, '/Material/SubjectGenderList995SubjectsAllRunsCompleted.mat' ) );
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
    
    save( strcat( OutputPath, '/Result_Sex.mat'), 'PredictedScores', 'MeanPredictedScores', 'PredictedLabel', 'SubjectGenderList', 'C', 'PredictiveAccuracy'  );
    
elseif ( Behavior_No == 60 ) 
    
    save( strcat( OutputPath, '/Result_Age.mat'), 'PredictedScores', 'MeanPredictedScores' ); 
    
end
    
    