
# Packages / libraries
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, explained_variance_score, confusion_matrix, accuracy_score, classification_report, log_loss
from math import sqrt

######################
# Oversample with SMOTE and random undersample for imbalanced dataset
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
#####################

pd.options.mode.chained_assignment = None  # default='warn'


# Loading dataset

raw_dataset = pd.read_csv(
    'https://raw.githubusercontent.com/FilRus-mkws/Przejsciowka_test/main/Fuel%20Economy.csv',
    low_memory=False)

dataset = raw_dataset.copy()

dataset = dataset[['Year', 'Class', 'Drive', 'Transmission',
       'Engine Cylinders', 'Engine Displacement', 'Turbocharger',
       'Supercharger', 'Fuel Type', 'Fuel Type 1', 'Fuel Type 2',
       'City MPG (FT1)', 'Highway MPG (FT1)', 
       'Start Stop Technology']]



################################################################################
#CREATING DATASETS
dataset_turb = dataset[['Year', 'Class', 'Drive', 'Transmission',
       'Engine Cylinders', 'Engine Displacement', 'Turbocharger',
       'Fuel Type', 'Fuel Type 1', 'City MPG (FT1)', 'Highway MPG (FT1)']]

dataset_super = dataset[['Year', 'Class', 'Drive', 'Transmission',
       'Engine Cylinders', 'Engine Displacement', 'Fuel Type 2',
       'Supercharger', 'Fuel Type', 'Fuel Type 1', 
       'City MPG (FT1)', 'Highway MPG (FT1)']]

dataset_SST = dataset[['Year', 'Class', 'Drive', 'Transmission',
       'Engine Cylinders', 'Engine Displacement', 'Fuel Type', 
       'Fuel Type 1', 'City MPG (FT1)', 'Highway MPG (FT1)', 
       'Start Stop Technology', 'Fuel Type 2']]
################################################################################

# Creating a new 0-1 y variable

dataset_turb['Turbocharger'][dataset_turb['Turbocharger'] == 'T'] = 1
dataset_turb['Turbocharger'] = dataset_turb['Turbocharger'].fillna(0)
dataset_turb = dataset_turb.dropna()

dataset_super['Supercharger'][dataset_super['Supercharger'] == 'S'] = 1
dataset_super['Supercharger'] = dataset_super['Supercharger'].fillna(0)
dataset_super = dataset_super.dropna()

dataset_SST['Start Stop Technology'][dataset_SST['Start Stop Technology'] == 'Y'] = 1
dataset_SST['Start Stop Technology'][dataset_SST['Start Stop Technology'] == 'N'] = 0
dataset_SST['Start Stop Technology'] = dataset_SST['Start Stop Technology'].fillna(0)
dataset_SST = dataset_SST.dropna()


# Creating Dummies

features = ['Class', 'Drive', 'Transmission','Fuel Type', 'Fuel Type 1', 'Fuel Type 2']
features_t = ['Class', 'Drive', 'Transmission','Fuel Type', 'Fuel Type 1']

dataset_turb = pd.get_dummies(dataset_turb, columns = features_t)

dataset_super = pd.get_dummies(dataset_super, columns = features)

dataset_SST = pd.get_dummies(dataset_SST, columns = features)


# Running Correlation

hm = dataset.corr()

mask = np.zeros_like(hm)
mask[np.triu_indices_from(mask)] = True

"""
# Draw the heatmap with the mask
plt.figure('Heatmap MPG')
plt.clf()
plt.yticks(va="center")
plt.xticks(rotation=360)
#Visualizing Correlation with a Heatmap
sns.heatmap(hm, mask=mask, square=True,
            annot = True, annot_kws={'size':40}, cmap="Blues")
"""

# Split the data into X & y

#TURBOCHARGER

X_turb = dataset_turb.drop('Turbocharger', axis = 1).values
y_turb = dataset_turb['Turbocharger']

#SUPERCHARGER

X_super = dataset_super.drop('Supercharger', axis = 1).values
y_super = dataset_super['Supercharger']

#START STOP TECHNOLOGY

X_SST = dataset_SST.drop('Start Stop Technology', axis = 1).values
y_SST = dataset_SST['Start Stop Technology']



# Hold-out validation

############# First one ###############

#TURBOCHARGER
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_turb, y_turb,
                                                    train_size = 0.7,
                                                    test_size=0.3, random_state=10)

# summarize class distribution
counter = Counter(y_train_t)
print(counter)
# define pipeline
over = SMOTE(sampling_strategy=0.2)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)
# transform the dataset
X_train_t, y_train_t = pipeline.fit_resample(X_train_t, y_train_t)
# summarize the new class distribution
counter = Counter(y_train_t)
print(counter)

#SUPERCHARGER
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_super, y_super,
                                                    train_size = 0.7,
                                                    test_size=0.3, random_state=10)

#START STOP TECHNOLOGY
X_train_SST, X_test_SST, y_train_SST, y_test_SST = train_test_split(X_SST, y_SST,
                                                    train_size = 0.7,
                                                    test_size=0.3, random_state=10)

############# Second one ##############

#TURBOCHARGER
X_train_t, X_valid_t, y_train_t, y_valid_t = train_test_split(X_train_t,
                                                      y_train_t,
                                                      train_size = 0.9,
                                                      test_size=0.1, random_state=10)

#SUPERCHARGER
X_train_s, X_valid_s, y_train_s, y_valid_s = train_test_split(X_train_s,
                                                      y_train_s,
                                                      train_size = 0.9,
                                                      test_size=0.1, random_state=10)

#START STOP TECHNOLOGY
X_train_SST, X_valid_SST, y_train_SST, y_valid_SST = train_test_split(X_train_SST,
                                                      y_train_SST,
                                                      train_size = 0.9,
                                                      test_size=0.1, random_state=10)

# Training model

#TURBOCHARGER
weights_t = {0:1, 1:1}
log_reg_t = LogisticRegression(random_state=10, 
                               solver = 'lbfgs',class_weight=weights_t, max_iter=300000)
log_reg_t.fit(X_train_t, y_train_t)

#SUPERCHARGER
weights_s = {0:1, 1:4.5}
log_reg_s = LogisticRegression(random_state=10,
                               solver = 'lbfgs',class_weight=weights_s, max_iter=300000)
log_reg_s.fit(X_train_s, y_train_s)

#START STOP TECHNOLOGY
weights_SST = {0:1, 1:6.5}
log_reg_SST = LogisticRegression(random_state=10, 
                                 solver = 'lbfgs',class_weight=weights_SST, max_iter=300000)
log_reg_SST.fit(X_train_SST, y_train_SST)


# predict - Predict class labels for samples in X

#TURBOCHARGER

log_reg_t.predict(X_train_t)
y_pred_t = log_reg_t.predict(X_train_t)

#SUPERCHARGER

log_reg_s.predict(X_train_s)
y_pred_s = log_reg_s.predict(X_train_s)

#START STOP TECHNOLOGY

log_reg_SST.predict(X_train_SST)
y_pred_SST = log_reg_SST.predict(X_train_SST)


# predict_proba - Probability estimates
pred_proba_t = log_reg_t.predict_proba(X_train_t)

pred_proba_s = log_reg_s.predict_proba(X_train_s)

pred_proba_SST = log_reg_SST.predict_proba(X_train_SST)


#TURBOCHARGER
# Accuracy on Train
print("The Training Accuracy of Turbocharger is: ", log_reg_t.score(X_train_t, y_train_t))

# Accuracy on Test
print("The Testing Accuracy of Turbocharger is: ", log_reg_t.score(X_test_t, y_test_t))

# Accuracy on Valid
print("The Validation Accuracy of Turbocharger is: ", log_reg_t.score(X_valid_t, y_valid_t))


#SUPERCHARGER
# Accuracy on Train
print("The Training Accuracy of Supercharger is: ", log_reg_s.score(X_train_s, y_train_s))

# Accuracy on Test
print("The Testing Accuracy of Supercharger is: ", log_reg_s.score(X_test_s, y_test_s))

# Accuracy on Valid
print("The Validation Accuracy of Supercharger is: ", log_reg_s.score(X_valid_s, y_valid_s))

#START STOP TECHNOLOGY
# Accuracy on Train
print("The Training Accuracy of SST is: ", log_reg_SST.score(X_train_SST, y_train_SST))

# Accuracy on Test
print("The Testing Accuracy of SST is: ", log_reg_SST.score(X_test_SST, y_test_SST))

# Accuracy on Valid
print("The Validation Accuracy of SST is: ", log_reg_SST.score(X_valid_SST, y_valid_SST))
print("")

# Classification Report

print("CLASSIFACATION RAPORT OF TURBOCHARGER (TRAINING)\n",  classification_report(y_train_t, 
                                                                      log_reg_t.predict(X_train_t)))
print("CLASSIFACATION RAPORT OF TURBOCHARGER (TESTING)\n",  classification_report(y_test_t, 
                                                                      log_reg_t.predict(X_test_t)))
print("CLASSIFACATION RAPORT OF TURBOCHARGER (VALIDATION)\n",  classification_report(y_valid_t, 
                                                                      log_reg_t.predict(X_valid_t)))


print("CLASSIFACATION RAPORT OF SUPERCHARGER (TRAINING)\n",  classification_report(y_train_s,
                                                                      log_reg_s.predict(X_train_s)))
print("CLASSIFACATION RAPORT OF SUPERCHARGER (TESTING)\n",  classification_report(y_test_s,
                                                                      log_reg_s.predict(X_test_s)))
print("CLASSIFACATION RAPORT OF SUPERCHARGER (VALIDATION)\n",  classification_report(y_valid_s,
                                                                      log_reg_s.predict(X_valid_s)))


print("CLASSIFACATION RAPORT OF SST (TRAINING)\n",  classification_report(y_train_SST,
                                                             log_reg_SST.predict(X_train_SST)))
print("CLASSIFACATION RAPORT OF SST (TESTING)\n",  classification_report(y_test_SST,
                                                             log_reg_SST.predict(X_test_SST)))
print("CLASSIFACATION RAPORT OF SST (VALIDATION)\n",  classification_report(y_valid_SST,
                                                             log_reg_SST.predict(X_valid_SST)))

# Confusion Matrix function

def plot_confusion_matrix(cm, classes=None, title='Confusion matrix'):
    """Plots a confusion matrix."""
    if classes is not None:
        sns.heatmap(cm, cmap="YlGnBu", xticklabels=classes, yticklabels=classes, 
                    vmin=0., vmax=1., annot=True, annot_kws={'size':30})
    else:
        sns.heatmap(cm, vmin=0., vmax=1.)
    plt.title(title)
    plt.ylabel('Klasa rzeczywista')
    plt.xlabel('Klasa predykowana')

# Visualizing cm

###  TURBOCHARGER  ####

plt.figure("Turbocharger")
plt.clf()
plt.suptitle('Tablica pomyłek - Turbodoładowanie', fontsize=15) 


cm_train = confusion_matrix(y_train_t, y_pred_t)
cm_norm_train = cm_train / cm_train.sum(axis=1).reshape(-1,1)

plt.subplot(131)
plot_confusion_matrix(cm_norm_train, classes = log_reg_t.classes_, title='Trening')

cm_test = confusion_matrix(y_test_t, log_reg_t.predict(X_test_t))
cm_norm_test = cm_test / cm_test.sum(axis=1).reshape(-1,1)

plt.subplot(132)
plot_confusion_matrix(cm_norm_test, classes = log_reg_t.classes_, title='Test')

cm_valid = confusion_matrix(y_valid_t, log_reg_t.predict(X_valid_t))
cm_norm_valid = cm_valid / cm_valid.sum(axis=1).reshape(-1,1)

plt.subplot(133)
plot_confusion_matrix(cm_norm_valid, classes = log_reg_t.classes_, title='Walidacja')

#plt.tight_layout()



###  SUPERCHARGER  ####

plt.figure("Supercharger")
plt.clf()
plt.suptitle('Tablica pomyłek - Sprężarka doładowująca', fontsize=15) 


cm_train = confusion_matrix(y_train_s, y_pred_s)
cm_norm_train = cm_train / cm_train.sum(axis=1).reshape(-1,1)

plt.subplot(131)
plot_confusion_matrix(cm_norm_train, classes = log_reg_s.classes_, title='Trening')

cm_test = confusion_matrix(y_test_s, log_reg_s.predict(X_test_s))
cm_norm_test = cm_test / cm_test.sum(axis=1).reshape(-1,1)

plt.subplot(132)
plot_confusion_matrix(cm_norm_test, classes = log_reg_s.classes_, title='Test')

cm_valid = confusion_matrix(y_valid_s, log_reg_s.predict(X_valid_s))
cm_norm_valid = cm_valid / cm_valid.sum(axis=1).reshape(-1,1)

plt.subplot(133)
plot_confusion_matrix(cm_norm_valid, classes = log_reg_s.classes_, title='Walidacja')

#plt.tight_layout()


###  SST  ####

plt.figure("SST")
plt.clf()
plt.suptitle('Tablica pomyłek - SST', fontsize=15) 


cm_train = confusion_matrix(y_train_SST, y_pred_SST)
cm_norm_train = cm_train / cm_train.sum(axis=1).reshape(-1,1)

plt.subplot(131)
plot_confusion_matrix(cm_norm_train, classes = log_reg_SST.classes_, title='Trening')

cm_test = confusion_matrix(y_test_SST, log_reg_SST.predict(X_test_SST))
cm_norm_test = cm_test / cm_test.sum(axis=1).reshape(-1,1)

plt.subplot(132)
plot_confusion_matrix(cm_norm_test, classes = log_reg_SST.classes_, title='Test')

cm_valid = confusion_matrix(y_valid_SST, log_reg_SST.predict(X_valid_SST))
cm_norm_valid = cm_valid / cm_valid.sum(axis=1).reshape(-1,1)

plt.subplot(133)
plot_confusion_matrix(cm_norm_valid, classes = log_reg_SST.classes_, title='Walidacja')

#plt.tight_layout()

