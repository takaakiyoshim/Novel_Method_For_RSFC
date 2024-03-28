# Novel method for resting-state functional connectivity (RSFC)

We developed novel method to estimmate human RSFC by using fine-grained local spatial pattens.
The ensemble learner genererated from RSFCs of both traditional and our novel methods acheived better predictive perfromance than single learner of the traditional RSFC. 

Ref : 

These matlab codes illustrate the procedures of our novel methods.
Main steps could be divided into 3 main steps as described bellow.

```
Step 1. Two types of the RSFCs are caculated 
simulateneously. 
```

```
Step 2. The SVM (Support vector machine) learner constructed from traditonal or novel RSFC seperately are used to predict human traits and behaviors.
```

```
Step 3. Ensemble learner construced from both two RSTFs is also used to predict human traits and behaviors.
```

## Code_01_make_RSFC_2Types.m
 Two types of the RSFCs are caculated simulateneously. These FCs are strored in suquare matrices form.  

## Code_02_make_RSFC_Vectorized_2Types.m
These matrices are vectorized and concatinated for each method.

## Code_03_predict_Trait_Behavior_By_SVM.m
SVM learner is used to predict redict human traits and behaviors for each method.
## Code_04_predict_Trait_Behavior_By_SVM_Nested_CV_For_Ensemble.m
Weight average techinque is adopted for ensemble learning method. To avoid data leakage, we caclculate predicted scores for each method by using the nested cross validation method.
## Code_05_predict_Trait_Behavior_By_Ensemble.m
Ensemble learer used to  predict redict human traits and behaviors by using Code 03 and 04 results.

## Directory structure containg results.
```
ResultData
├── FC_Matrix_Novel_Method
├── FC_Matrix_Traditional_Method
├── Predicted_Score_Ensemble
└── Predicted_Score_Nested_CV_For_Ensemble
    └── Novel_Method
        ├── Predicted_Score_Age
        ├── Predicted_Score_Behavior_No1
        ･････････
        ├── Predicted_Score_Behavior_No58
        ├── Predicted_Score_Sex
    └── Traditional_Method
        ├── Predicted_Score_Age
        ├── Predicted_Score_Behavior_No1
        ･････････
        ├── Predicted_Score_Behavior_No58
        ├── Predicted_Score_Sex
├─Predicted_Score_Single_Method
│  ├─Novel_Method
│  └─Traditional_Method
└─Vectorized_FC
```


