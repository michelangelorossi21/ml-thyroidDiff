import statistics


from colorama import Fore
import pandas as pd
import numpy as np
import seaborn as sea
import warnings
import argparse
from colorama import Fore

from sklearn import metrics
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def read_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset_path', type=str,
                        default='Thyroid_Diff.csv',
                        help='Path to the file containing the dataset.')

    args = parser.parse_args()

    return args


def print_info(df:pd.DataFrame):
    print(Fore.YELLOW + '\nDataset info:', Fore.RESET)

    print(df.info)
    print(df.describe())
    print(df.head())
    print(df.columns)


def exploratory_data_analysis(df: pd.DataFrame):

    # Check for missing values:
    print(Fore.GREEN + "Missing values: " + Fore.RESET)
    print(df.isnull().any())  # there are no missing values in any column

    # Check for num or cat features:
    print(Fore.GREEN + "\nData types: " + Fore.RESET)
    print(df.dtypes)  # Age: integer, every other feature: cat

    # Split between X and y:
    y = df['Recurred']
    age = df['Age']
    X = df.drop(['Age', 'Recurred'], axis=1)  # da X tolgo la var. target e l'unica feature numerica

    # Discretize categorical cols:
    y, _ = pd.factorize(df['Recurred'], sort=True)

    for col in X.columns:
        X[col], _ = pd.factorize(X[col], sort=True)

    # Cast age to float64:
    age = age.values.astype('float64')

    X.insert(loc=0, column='Age', value=age)  # ri-inserisco in X la feature Age, l'unica numerica

    # Split between train and test:
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    return X_train, y_train, X_test, y_test


def create_models():

    models = [
        KNeighborsClassifier(weights='distance'),
        LogisticRegression(solver='saga', class_weight='balanced'),
        SVC(class_weight='balanced'),
        DecisionTreeClassifier(class_weight='balanced'),
    ]

    models_names = ['K-NN', 'Logistic Reg.', 'SVM', 'Decision Tree']

    models_hparameters = [
        {'n_neighbors': list(range(1, 10, 2))},  # K-Nearest Neighbours
        {'penalty': ['l1', 'l2'], 'C': [1e-5, 5e-5, 1e-4, 5e-4, 1]},  # Logistic Regression
        {'C': [1e-4, 1e-2, 1, 1e1, 1e2], 'gamma': [0.001, 0.0001],  # SVM
         'kernel': ['linear', 'poly', 'rbf']},
        {'criterion': ['gini', 'entropy']}  # Decision Tree
    ]

    print("\n...Finished.")
    return models, models_names, models_hparameters


def hparameters_tuning(X_train, y_train, models, models_name, models_hparameters):

    estimators = []
    chosen_hparameters = []

    # Hyperparameters tuning:
    for model, model_name, model_hparameters in zip(models, models_name, models_hparameters):
        print('\nModel name: ', model_name)

        clf = GridSearchCV(estimator=model, param_grid=model_hparameters, scoring='accuracy', cv=5)
        clf.fit(X_train, y_train)

        chosen_hparameters.append(clf.best_params_)
        estimators.append((clf.best_score_, model_name, clf))

        print('Accuracy: ', clf.best_score_)
        for hparam in model_hparameters:
            print(f'\t The best choice for parameter {hparam}: ', clf.best_params_.get(hparam))

    # Sort estimators by scores:
    estimators = sorted(estimators, reverse=True)
    return estimators, chosen_hparameters


def plot_results(y_test, y_pred):
    fig = plt.figure()
    plt.hist(y_test, align='left')
    plt.hist(y_pred, align='right')
    plt.legend(['y_test', 'y_pred'])
    plt.xticks((0, 1))
    plt.show()


if __name__ == '__main__':

    args = read_args()

    df = pd.read_csv(args.dataset_path, delimiter=',')
    print_info(df)

    # 1. Exploratory Data Analysis
    print(Fore.YELLOW + '\n1. Exploratory Data Analysis', Fore.RESET)
    X_train, y_train, X_test, y_test = exploratory_data_analysis(df)

    # Scalamento dei dati:
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)

    # 2. Creare i modelli (K-NN, Logistic Regression, SVM, Decision Tree)
    print(Fore.YELLOW, '\n2. Create models', Fore.RESET)
    models, models_name, models_hparameters = create_models()

    # 3a. Tuning degli iperparametri, istanziare i modelli
    print(Fore.YELLOW, '\n3. Tuning hyperparameters for each model....', Fore.RESET)
    estimators, chosen_hparameters = hparameters_tuning(X_train, y_train, models, models_name, models_hparameters)

    # 3b. Scegliere il modello migliore: Decision Tree
    final_model = estimators[0][2]
    print("\nThe best model is: ", estimators[0][1])

    # 4. Training e Cross-Validation:
    print(Fore.YELLOW, '\n4. Cross-Validation and Training...', Fore.RESET)
    scores = cross_validate(estimator=final_model, X=X_train, y=y_train, cv=5,
                            scoring=('accuracy', 'f1_weighted'))

    print('\nThe Accuracy of the final model is: ', np.mean(scores['test_accuracy']))
    print('The F1-score of the final model is: ', np.mean(scores['test_f1_weighted']))

    # Wrapper:
    sfs = SequentialFeatureSelector(final_model)
    sfs.fit(X_train, y_train)
    X_train = sfs.transform(X_train)
    print('\nNumber of selected features: ', sfs.get_support()[sfs.get_support()].size)

    final_model.fit(X_train, y_train)

    # 5. Testing e risultati finali:
    print(Fore.YELLOW, '\n5. Testing....', Fore.RESET)

    X_test = scaler.transform(X_test)
    X_test = sfs.transform(X_test)
    y_pred = final_model.predict(X_test)

    print("\n...Finished.")

    # Plotting results:
    plot_results(y_test, y_pred)

    # Metrics:
    target_names = ['No', 'Yes']

    print(Fore.YELLOW, '\n6. Final results:', Fore.RESET)
    print(metrics.classification_report(y_test, y_pred, target_names=target_names))
    print('Accuracy is: ', accuracy_score(y_test, y_pred))
    print('Precision is: ', precision_score(y_test, y_pred, average='weighted'))
    print('Recall is: ', recall_score(y_test, y_pred, average='weighted'))
    print('F1-score is: ', f1_score(y_test, y_pred, average='weighted'))

