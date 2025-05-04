import pandas as pd
import numpy as np
from datetime import datetime
import os
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score, log_loss, roc_curve
import matplotlib.pyplot as plt

def convert_decimal_to_double(df):
    """Convert decimal columns to double type in a Spark DataFrame"""
    from pyspark.sql.types import DecimalType, DoubleType
    
    for field in df.schema.fields:
        if isinstance(field.dataType, DecimalType):
            df = df.withColumn(field.name, df[field.name].cast(DoubleType()))
    return df

def create_directory(project_name=None, volume_name=None):
    """Create a directory for storing results
    
    Parameters:
    -----------
    project_name : str
        Name of the project
    volume_name : str
        Name of the volume
        
    Returns:
    --------
    str
        Path to the created directory
    """
    today = datetime.now()
    h = str(today.hour)
    time_str = today.strftime('%d') + 'th_' + h + ':' + str(today.minute) + '_min'
    
    if project_name is None:
        project_name = str(input('PLEASE ENTER PROJECT NAME : '))
    
    if volume_name is None:
        volume_name = str(input('ENTER THE VOLUME NAME TO STORE CSV : '))
    
    try:
        # For Databricks environments
        try:
            from pyspark.dbutils import DBUtils
            spark.sql(f'CREATE VOLUME IF NOT EXISTS data_lake.analytics.{volume_name}')
            dbfs_mount_point = f"/Volumes/data_lake/analytics/{volume_name}/{project_name}_{time_str}"
        except:
            # For local environments
            dbfs_mount_point = f"./output/{project_name}_{time_str}"
        
        os.makedirs(dbfs_mount_point, mode=0o777, exist_ok=False)
    except:
        try:
            # Try another location if first attempt fails
            dbfs_mount_point = f"./output/{project_name}_{time_str}"
            os.makedirs(dbfs_mount_point, mode=0o777, exist_ok=False)
        except:
            # Use a timestamp if both attempts fail
            dbfs_mount_point = f"./output/scorecard_{int(datetime.now().timestamp())}"
            os.makedirs(dbfs_mount_point, mode=0o777, exist_ok=True)
    
    return dbfs_mount_point

def split_numerical_categorical(data, exclude_cols=None):
    """Split data into numerical and categorical columns
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
    exclude_cols : list
        Columns to exclude from the split
        
    Returns:
    --------
    tuple
        List of numerical columns and list of categorical columns
    """
    if exclude_cols is None:
        exclude_cols = []
    
    numerical_cols = data.select_dtypes(exclude='object').columns
    numerical_cols_list = [col for col in numerical_cols if col not in exclude_cols]
    
    categorical_cols = data.select_dtypes(include='object').columns
    categorical_cols_list = [col for col in categorical_cols if col not in exclude_cols]
    
    return numerical_cols_list, categorical_cols_list

def filter_by_variance(data, numerical_cols, threshold=0.02):
    """Filter out low-variance features
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
    numerical_cols : list
        List of numerical columns
    threshold : float
        Variance threshold
        
    Returns:
    --------
    list
        List of removed features
    """
    sel = VarianceThreshold(threshold=threshold)
    sel.fit(data[numerical_cols].fillna(0))
    
    removed_features = [col for col in numerical_cols 
                      if col not in data[numerical_cols].columns[sel.get_support()]]
    
    return removed_features

def plot_roc_curve(y_true, p_base_learners, p_ensemble, labels, ens_label):
    """Plot ROC curves for base learners and ensemble
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    p_base_learners : array-like
        Predictions of base learners
    p_ensemble : array-like
        Predictions of ensemble model
    labels : list
        Names of base learners
    ens_label : str
        Name of ensemble model
    """
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--')
    
    # Generate colors
    cm = [plt.cm.rainbow(i) for i in np.linspace(0, 1.0, p_base_learners.shape[1] + 1)]
    
    # Plot ROC curves for base learners
    for i in range(p_base_learners.shape[1]):
        p = p_base_learners[:, i]
        fpr, tpr, _ = roc_curve(y_true, p)
        plt.plot(fpr, tpr, label=labels[i], c=cm[i + 1])
    
    # Plot ROC curve for ensemble
    fpr, tpr, _ = roc_curve(y_true, p_ensemble)
    plt.plot(fpr, tpr, label=ens_label, c=cm[0])
    
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(frameon=False)
    plt.show() 