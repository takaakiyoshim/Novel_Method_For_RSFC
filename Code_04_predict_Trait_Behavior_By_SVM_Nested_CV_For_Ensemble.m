function Code_04_predict_Trait_Behavior_By_SVM_Nested_CV_For_Ensemble(method_name,Behavior_No,iter_CV_No,PWD)


Behavior_No = str2num(Behavior_No);
iter_CV_No = str2num(iter_CV_No);

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
if ( Behavior_No < 59 )
  OutputPath = strcat( PWD, '/ResultData/Predicted_Score_Nested_CV_For_Ensemble/', method_name, '_Method/Predicted_Score_Behavior_No', num2str(Behavior_No) );
elseif ( Behavior_No == 59 )
  OutputPath = strcat( PWD, '/ResultData/Predicted_Score_Nested_CV_For_Ensemble/', method_name, '_Method/Predicted_Score_Sex' );
elseif ( Behavior_No == 60 )
  OutputPath = strcat( PWD, '/ResultData/Predicted_Score_Nested_CV_For_Ensemble/', method_name, '_Method/Predicted_Score_Age' );
end


%% make result container
PredictedScores = NaN(995*4,1);


%% outer loop : 10 fold cross validation 
%% innner loop : 9 fold cross validation
if ( Behavior_No ~= 59 )
   %% solve regression problem
    %% 10 fold cross validation 
    %% using fitrsvm 
 
  
        %% set cv %%
        TrainCV = CVpartion{iter_CV_No,1};
        TestCV = CVpartion{iter_CV_No,2};

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

        PredictedScores = NaN(995*4,1);

        for iterInCV = 1:10
           
            if ( isequal( iter_CV_No, iterInCV ) == 0 )

                %% set inner cv
                InTestCV = CVpartion{iterInCV,2};
                InTestCV_Multiplied = [];
                for s = 1:length(InTestCV)
                    InTestCV_Multiplied = cat(1,InTestCV_Multiplied,[(4*InTestCV(s,1)-3):4*InTestCV(s,1)]');
                end
                clear s;
                InTrainCV_Multiplied = TrainCV_Multiplied;

                for s = 1:length(InTestCV_Multiplied)
                    InTrainCV_Multiplied(find(InTrainCV_Multiplied==InTestCV_Multiplied(s))) = [];
                end
                clear s;

                %% divide X, Y
                X_Training = X(:,InTrainCV_Multiplied);
                X_Test = X(:,InTestCV_Multiplied);
                Y_Training = Y_Multiplied(InTrainCV_Multiplied);

                %% model fitting
                Mdl = fitrsvm(X_Training',Y_Training);
                %% estimation
                Score = predict(Mdl,X_Test');

                PredictedScores(InTestCV_Multiplied) = Score;
            end
            
        end

     
elseif ( Behavior_No == 59 )
    
     %% solve classification problem for sex difference
    %% 10 fold cross validation 
    %% using fitcsvm function
    
     %% set cv %%
    TrainCV = CVpartion{iter_CV_No,1};
    TestCV = CVpartion{iter_CV_No,2};

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
    
    PredictedScores = NaN(995*4,1);
    
    for iterInCV = 1:10
        tic
        if ( isequal( iter_CV_No, iterInCV ) == 0 )

            %% set inner cv
            InTestCV = CVpartion{iterInCV,2};
            InTestCV_Multiplied = [];
            for s = 1:length(InTestCV)
                InTestCV_Multiplied = cat(1,InTestCV_Multiplied,[(4*InTestCV(s,1)-3):4*InTestCV(s,1)]');
            end
            clear s;
            InTrainCV_Multiplied = TrainCV_Multiplied;
            
            for s = 1:length(InTestCV_Multiplied)
                InTrainCV_Multiplied(find(InTrainCV_Multiplied==InTestCV_Multiplied(s))) = [];
            end
            clear s;
            
            %% divide X, Y
            X_Training = X(:,InTrainCV_Multiplied);
            X_Test = X(:,InTestCV_Multiplied);
            
            Y_Training = cell(1,length(InTrainCV_Multiplied));
            for s = 1:length(InTrainCV_Multiplied)
                Y_Training{1,s} = Y_Multiplied{1,InTrainCV_Multiplied(s)};
            end
            clear s;

            %% model fitting
            Mdl = fitcsvm(X_Training',Y_Training);
            %% estimation
            [~,Score] = predict(Mdl,X_Test');

            PredictedScores(InTestCV_Multiplied) = Score(:,1);
        end    
    end
end

%% save data
PredictedScores = PredictedScores(~isnan(PredictedScores));
save( strcat( OutputPath, 'PredictedScores_CV_No', num2str(iter_CV_No), '.mat' ), 'PredictedScores');



