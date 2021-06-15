# Predicting Credit Risk
## Overview
### *Purpose*
LendingClub, a peer-to-peer lending services company, wants to use machine learning to predict credit risk. The lead Data Scientist, Jill, asked for assistance in building several machine learning models to predict credit risk. The performance of these models were to be evaluated to determine whether they should be used to predict credit risk.

## Results
### *Random Oversampler*
The training data was resampled with the RandomOverSampler algorithm:
```
# Resample the training data with the RandomOversampler
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=1)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
Counter(y_resampled)
```
After training the logistic regression model using the resampled data, the balanced accuracy score was calculated:
```
# Calculated the balanced accuracy score
from sklearn.metrics import balanced_accuracy_score
y_pred = model.predict(X_test)
balanced_accuracy_score(y_test, y_pred)
```

![RandomOversampler_accScore.png]( https://github.com/kcharb7/Credit_Risk_Analysis/blob/main/Images/RandomOversampler_accScore.png)

-	The accuracy score of the Random Oversampler model was moderate at 0.640, indicating that the model was correct 64.0% of the time

Then, the classification report was printed:
```
# Print the imbalanced classification report
from imblearn.metrics import classification_report_imbalanced
print(classification_report_imbalanced(y_test, y_pred))
```
![RandomOversampler.png]( https://github.com/kcharb7/Credit_Risk_Analysis/blob/main/Images/RandomOversampler.png)

-	The precision for the high-risk loans is extremely low at 0.01, indicating a large number of false positives 
-	The recall for high-risk loans is 0.66, indicating a moderate number of false negatives
-	The F1 score is extremely low for high-risk loans at 0.02
-	Taken together, the Random Oversampler model is poor at classifying high-risk loans

### *SMOTE*
The training data was resampled with the SMOTE algorithm:
```
# Resample the training data with SMOTE
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE(random_state=1,
sampling_strategy='auto').fit_resample(
   X_train, y_train)
Counter(y_resampled)
```
After training the logistic regression model using the resampled data, the balanced accuracy score was calculated:
```
# Calculated the balanced accuracy score
from sklearn.metrics import balanced_accuracy_score
y_pred = model.predict(X_test)
balanced_accuracy_score(y_test, y_pred)
```

![SMOTE_accScore.png]( https://github.com/kcharb7/Credit_Risk_Analysis/blob/main/Images/SMOTE_accScore.png)

-	The accuracy score of the SMOTE model was moderate at 0.651, indicating that the model was correct 65.1% of the time

Then, the classification report was printed:
```
# Print the imbalanced classification report
from imblearn.metrics import classification_report_imbalanced
print(classification_report_imbalanced(y_test, y_pred))
```

![SMOTE.png]( https://github.com/kcharb7/Credit_Risk_Analysis/blob/main/Images/SMOTE.png)

-	The precision for the high-risk loans is extremely low at 0.01, indicating a large number of false positives 
-	The recall for high-risk loans is 0.61, indicating a moderate number of false negatives
-	The F1 score is extremely low for high-risk loans at 0.02
-	Taken together, the SMOTE model is poor at classifying high-risk loans


### *Cluster Centroids*
The training data was resampled with the ClusterCentroids algorithm:
```
# Resample the data using the ClusterCentroids resampler
from imblearn.under_sampling import ClusterCentroids
cc = ClusterCentroids(random_state=1)
X_resampled, y_resampled = cc.fit_resample(X_train, y_train)
Counter(y_resampled)
```
After training the logistic regression model using the resampled data, the balanced accuracy score was calculated:
```
# Calculated the balanced accuracy score
from sklearn.metrics import balanced_accuracy_score
y_pred = model.predict(X_test)
balanced_accuracy_score(y_test, y_pred)
```
![Centroids_accScore.png]( https://github.com/kcharb7/Credit_Risk_Analysis/blob/main/Images/Centroids_accScore.png)

-	The accuracy score of the Cluster Centroids model was moderate at 0.544, indicating that the model was correct 54.4% of the time

Then, the classification report was printed:
```
# Print the imbalanced classification report
from imblearn.metrics import classification_report_imbalanced
print(classification_report_imbalanced(y_test, y_pred))
```

![Centroids.png]( https://github.com/kcharb7/Credit_Risk_Analysis/blob/main/Images/Centroids.png)

-	The precision for the high-risk loans is extremely low at 0.01, indicating a large number of false positives 
-	The recall for high-risk loans is 0.69, indicating a moderate number of false negatives
-	The F1 score is extremely low for high-risk loans at 0.01
-	Taken together, the Cluster Centroids model is poor at classifying high-risk loans


### *SMOTEENN*
The training data was resampled with SMOTEENN:
```
# Resample the training data with SMOTEENN
from imblearn.combine import SMOTEENN
smote_enn = SMOTEENN(random_state=0)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)
Counter(y_resampled)
```
After training the logistic regression model using the resampled data, the balanced accuracy score was calculated:
```
# Calculated the balanced accuracy score
from sklearn.metrics import balanced_accuracy_score
y_pred = model.predict(X_test)
balanced_accuracy_score(y_test, y_pred)
```

![SMOTEENN_accScore.png]( https://github.com/kcharb7/Credit_Risk_Analysis/blob/main/Images/SMOTEENN_accScore.png)

-	The accuracy score of the SMOTEENN model was low at 0.645, indicating that the model was correct 64.5% of the time

Then, the classification report was printed:
```
# Print the imbalanced classification report
from imblearn.metrics import classification_report_imbalanced
print(classification_report_imbalanced(y_test, y_pred))
```

![SMOTEENN.png]( https://github.com/kcharb7/Credit_Risk_Analysis/blob/main/Images/SMOTEENN.png)

-	The precision for the high-risk loans is extremely low at 0.01, indicating a large number of false positives 
-	The recall for high-risk loans is 0.72, indicating a moderate number of false negatives
-	The F1 score is extremely low for high-risk loans at 0.02
-	Taken together, the SMOTEENN model is moderate at classifying high-risk loans

### *Balanced Random Forest Classifier*
The training data was resampled with the BalancedRandomForestClassifier:
```
# Resample the training data with the BalancedRandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
balanced_rf_model = BalancedRandomForestClassifier(n_estimators=100, random_state=1)
balanced_rf_model = balanced_rf_model.fit(X_train, y_train)
```
After training the logistic regression model using the resampled data, the balanced accuracy score was calculated:
```
# Calculated the balanced accuracy score
from sklearn.metrics import balanced_accuracy_score
y_pred = model.predict(X_test)
balanced_accuracy_score(y_test, y_pred)
```
![Balanced_rf_accScore.png]( https://github.com/kcharb7/Credit_Risk_Analysis/blob/main/Images/Balanced_rf_accScore.png)

-	The accuracy score of the Balanced Random Forest Classifier model was moderate at 0.789, indicating that the model was correct 78.9% of the time

Then, the classification report was printed:
```
# Print the imbalanced classification report
from imblearn.metrics import classification_report_imbalanced
print(classification_report_imbalanced(y_test, y_pred))
```

![ Balanced_rf.png]( https://github.com/kcharb7/Credit_Risk_Analysis/blob/main/Images/Balanced_rf.png)

-	The precision for the high-risk loans is extremely low at 0.03, indicating a large number of false positives 
-	The recall for high-risk loans is 0.70, indicating a moderate number of false negatives
-	The F1 score is extremely low for high-risk loans at 0.06
-	Taken together, the Balanced Random Forest Classifier model is moderate at classifying high-risk loans

### *Easy Ensemble AdaBoost Classifier*
The data was trained with the EasyEnsembleClassifier:
```
# Train the EasyEnsembleClassifier
from imblearn.ensemble import EasyEnsembleClassifier
easy_ensemble = EasyEnsembleClassifier(n_estimators=100, random_state=1)
easy_ensemble = easy_ensemble.fit(X_train, y_train)
```
After training the logistic regression model using the resampled data, the balanced accuracy score was calculated:
```
# Calculated the balanced accuracy score
from sklearn.metrics import balanced_accuracy_score
y_pred = model.predict(X_test)
balanced_accuracy_score(y_test, y_pred)
```
![Ensemble_accScore.png]( https://github.com/kcharb7/Credit_Risk_Analysis/blob/main/Images/Ensemble_accScore.png)

-	The accuracy score of the Easy Ensemble AdaBoost Classifier model was high at 0.932, indicating that the model was correct 93.2% of the time

Then, the classification report was printed:
```
# Print the imbalanced classification report
from imblearn.metrics import classification_report_imbalanced
print(classification_report_imbalanced(y_test, y_pred))
```

![ Ensemble_rf.png]( https://github.com/kcharb7/Credit_Risk_Analysis/blob/main/Images/Ensemble.png)

-	The precision for the high-risk loans is extremely low at 0.09, indicating a large number of false positives 
-	The recall for high-risk loans was high at 0.92, indicating a low number of false negatives
-	The F1 score was low for high-risk loans at 0.16
-	Taken together, the Easy Ensemble AdaBoost Classifier model is high at classifying high-risk loans

## Summary
In summary, six machine learning models were built and evaluated for predicting credit risk. The first three models used resampling, and included the oversampling RandomOverSampler and SMOTE algorithms, and the undersampling ClusterCentroids algorithm. Each of the oversampling models had moderate accuracy scores around 65%, while the undersampling model had the lowest accuracy score at 54.4%. All resampling models had extremely low precision scores at 0.01 and moderate recall scores around 0.6. The fourth model tested was the SMOTEENN algorithm that additionally used resampling through a combinatorial approach of over- and undersampling. This model had an accuracy score of 64.5%, a precision score of 0.01, and a recall score of 0.72. The final two models were ensemble classifiers and had higher accuracy scores among the models at 78.9% for the Balanced Random Forest Classifier and 93.2% for the Easy Ensemble AdaBoost Classifier. These models also had higher recall scores at 0.70 and 0.92, respectively. 

As the Ensemble AdaBoost Classifier model had the highest accuracy score at 93.2% and the highest recall at 0.92, it is recommended that this model be used to predict credit risk. While the precision for high-risk loans is low, having an extremely high recall is more important when determining high-risk loans. It is better for a company to have less false negatives than false positives for high-risk loans as a false negative would result in the bank providing a loan to an individual who they believed to be low-risk when they are actually high-risk. Such an action could be detrimental to the lending company.
