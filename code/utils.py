import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np
import xgboost as xgb
from sklearn import tree
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import clone
from utils import *

def plot_histogram(df, colname, data_dict):
    # setting variables
    desc = data_dict[colname]['var_desc_short']
    vals = data_dict[colname]['data_values']
    # plotting figure
    plt.figure(figsize=(10,5))
    sns.histplot(df[colname])
    plt.xlabel(f'Value of {colname} ({vals})')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of values in {colname} ({desc})')
    plt.savefig(f'../img/eda/{colname}_hist.jpeg',dpi=900)
    plt.show()

def save_results_to_df(df, y_test, y_pred, model_name):
    # calculating metrics
    model_accuracy = accuracy_score(y_test, y_pred)
    model_precision = precision_score(y_test, y_pred)
    model_recall = recall_score(y_test, y_pred)
    model_f1_score = f1_score(y_test, y_pred)
    # obtaining confusion matrix
    model_confusion = confusion_matrix(y_test, y_pred)
    # saving results to df
    df.loc[len(df)] = [model_name, model_accuracy, model_precision, model_recall, model_f1_score]
    return df, model_confusion

def plot_confusion_matrices(df, classifiers, train_or_test):
    # Create subplots
    fig, ax = plt.subplots(2, len(classifiers)//2, sharex=True, sharey=False, figsize=(20, 12), dpi=300)
    fig.suptitle(f"Confusion Matrices for {train_or_test} Results", fontsize=16)

    # Add an axis for the colorbar
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

    for i, classifier in enumerate(classifiers):
        # Calculate subplot indices
        row = i // (len(classifiers)//2)
        col = i % (len(classifiers)//2)

        # Plot for train results without individual colorbars
        ConfusionMatrixDisplay(df[i], display_labels=[False, True]).plot(ax=ax[row, col], cmap='RdPu', colorbar=False)
        ax[row, col].set_title(f'{classifier}')
        ax[row, col].tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better visibility

    # Add a single colorbar for all subplots
    fig.colorbar(ax[-1, -1].get_images()[0], cax=cax, orientation='vertical', label='Count')

    # Adjust layout
    fig.tight_layout(rect=[0, 0.03, 0.9, 0.95])

    # Show the plot
    plt.show()

def custom_xgb_fit(model, X, y, sample_weight=None, base_margin=None, eval_set=None, eval_metric=None,
                   early_stopping_rounds=None, verbose=True, xgb_model=None, sample_weight_eval_set=None):
    return model.fit(X, y)

def create_feature_importances_df(model, X, y, data_dict, n_repeats=30, random_state=42):
    # If the model has 'feature_importances_' attribute, use it directly
    if hasattr(model, 'feature_importances_'):
        feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    else:
        # Clone the model to avoid modifying the original
        cloned_model = clone(model)
        # Use the custom fit function for XGBoost
        cloned_model.fit = lambda X, y: custom_xgb_fit(cloned_model, X, y)
        # Run permutation importance
        permutation_importances = permutation_importance(cloned_model, X, y,
                                                         n_repeats=n_repeats,
                                                         random_state=random_state)
        feature_importances = pd.Series(permutation_importances.importances_mean, index=X.columns)

    # Sort and create the DataFrame as before
    feature_importances = feature_importances.sort_values(ascending=False)
    feature_descriptions = {var: data_dict[var]['var_desc_short'] for var in feature_importances.index}

    df = pd.DataFrame({
        'Variable Name': feature_importances.index,
        'Short Description': [feature_descriptions[var] for var in feature_importances.index],
        'Importance': feature_importances.values
    })

    return df

def visualize_feature_importance(feature_importances_df, model_name):
    """
    Visualize feature importance using a horizontal bar plot with Matplotlib.

    Parameters:
    - feature_importances_df: DataFrame with columns 'Variable Name', 'Short Description', and 'Importance'
    - model_name: Name of the model for plot title
    """
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Variable Name', data=feature_importances_df, palette='bwr')
    plt.title(f'Feature Importance - {model_name}')
    plt.xlabel('Importance')
    plt.ylabel('Variable Name')
    plt.show()

# Function to create a DataFrame with feature importances and variable descriptions for Logistic Regression
def create_feature_importances_df_lr(model, data_dict, lr_feature_importances):
    # Map variable descriptions to feature names
    feature_descriptions = {var: data_dict[var]['var_desc_short'] for var in lr_feature_importances.index}
    # Create a DataFrame
    df = pd.DataFrame({
        'Variable Name': lr_feature_importances.index,
        'Short Description': [feature_descriptions[var] for var in lr_feature_importances.index],
        'Importance': lr_feature_importances.values
    })
    return df

def create_feature_importances_df_knn(model, X, y, data_dict, n_repeats=30, random_state=None):
    # Fit the model
    model.fit(X, y)
    
    # Calculate permutation importances with parallelization
    permutation_importances = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
    
    # Create a DataFrame with feature importances and variable descriptions
    feature_importances_df = pd.DataFrame({'Importance': permutation_importances.importances_mean}, index=X.columns)
    
    # Sort feature importances in descending order
    feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)
    
    # Add the variable name column
    feature_importances_df['Variable Name'] = feature_importances_df.index
    feature_importances_df.reset_index(drop=True, inplace=True)
    
    # Map variable descriptions to feature names
    feature_importances_df['Short Description'] = feature_importances_df['Variable Name'].map(data_dict)
    
    # Extract the values from the dictionary in 'Short Description'
    feature_importances_df['Short Description'] = feature_importances_df['Short Description'].apply(lambda x: x.get('var_desc_short', ''))
    
    # Reorder columns
    feature_importances_df = feature_importances_df[['Variable Name', 'Short Description', 'Importance']]
    
    return feature_importances_df