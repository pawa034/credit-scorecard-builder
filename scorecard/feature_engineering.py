import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from optbinning import OptimalBinning
from sklearn.inspection import permutation_importance
from lightgbm import LGBMClassifier
from probatus.feature_elimination import ShapRFECV


def iv_maker(uid, target, data, variable=None):
    """Calculate Information Value (IV) for specified variables
    
    Parameters:
    -----------
    uid : str
        Unique identifier column name
    target : str
        Name of the target variable
    data : pd.DataFrame
        DataFrame containing the dataset
    variable : list or None
        List of variable names to calculate IV for
        
    Returns:
    --------
    dict
        Dictionary containing variable names and corresponding binning objects
    """
    try:
        # Default parameters
        min_n_bins = 2
        max_n_bins = 10
        min_bin_size = 0.05
        monotonic_trend = "auto_asc_desc"
        min_event_rate_diff = 0
        special_values = [-999, -888]
        fine_binning_bad_fraction = 0.10
        cat_cutoff = 0.01

        # Validate target column
        if target not in data.columns:
            raise ValueError(f"The specified target column '{target}' is not found in the DataFrame.")

        # Set default bad rate difference based on data
        total_count = data.shape[0]
        bad_count = data[data[target] == 1].shape[0]
        bad_rate = bad_count / total_count
        min_event_rate_diff = bad_rate * fine_binning_bad_fraction

        # Get variables to process
        if variable is None:
            variables = [col for col in data.columns if col not in [uid, target]]
        else:
            variables = variable

        # Initialize result dictionary
        object_dict = {}

        # Identify numerical and categorical variables
        col1 = data[variables]._get_numeric_data().columns
        categorical_vars = set(variables) - set(col1)
        numerical_vars = set(variables) - set(categorical_vars)

        # Calculate IV for each variable
        for var in variables:
            try:
                if var in numerical_vars:
                    object_dict[var] = fit_numeric(
                        data, var, min_n_bins, max_n_bins,
                        min_bin_size, monotonic_trend,
                        min_event_rate_diff, special_values, target
                    )
                elif var in categorical_vars:
                    object_dict[var] = fit_categorical(
                        data, var, min_n_bins,
                        special_values, fine_binning_bad_fraction, target,
                        min_bin_size, max_n_bins
                    )
                else:
                    print(f"Variable '{var}' is not numeric or categorical and will be skipped.")
            except Exception as e:
                print(f"An error occurred while processing variable '{var}': {str(e)}")

        return object_dict

    except Exception as e:
        print(f"An error occurred in iv_maker: {str(e)}")
        return None


def fit_numeric(data, var, min_n_bins, max_n_bins, min_bin_size, 
                monotonic_trend, min_event_rate_diff, special_values, target):
    """Fit a numerical variable using the OptimalBinning algorithm
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the dataset
    var : str
        Name of the numerical variable to be binned
    min_n_bins : int
        Minimum number of bins
    max_n_bins : int
        Maximum number of bins
    min_bin_size : float
        Minimum bin size
    monotonic_trend : str
        Desired monotonic trend
    min_event_rate_diff : float
        Minimum event rate difference
    special_values : list
        List of special values
    target : str
        Name of the target variable
        
    Returns:
    --------
    optbinning.binning.optimal_binning.OptimalBinning
        Fitted instance of the OptimalBinning algorithm
    """
    x = data[var].values
    y = data[target].values
    optb_numeric = OptimalBinning(
        name=var, 
        dtype="numerical", 
        min_n_bins=min_n_bins,
        max_n_bins=max_n_bins, 
        min_bin_size=min_bin_size, 
        monotonic_trend=monotonic_trend, 
        min_event_rate_diff=min_event_rate_diff, 
        special_codes=special_values
    )
    return optb_numeric.fit(x, y)


def fit_categorical(data, var, min_n_bins, special_values, cat_cutoff, 
                    target, min_bin_size, max_n_bins):
    """Fit a categorical variable using the OptimalBinning algorithm
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the dataset
    var : str
        Name of the categorical variable to be binned
    min_n_bins : int
        Minimum number of bins
    special_values : list
        List of special values
    cat_cutoff : float
        Category cutoff
    target : str
        Name of the target variable
    min_bin_size : float
        Minimum bin size
    max_n_bins : int
        Maximum number of bins
        
    Returns:
    --------
    optbinning.binning.optimal_binning.OptimalBinning
        Fitted instance of the OptimalBinning algorithm
    """
    x = data[var].values
    y = data[target].values
    special_list = [str(x) for x in special_values]
    optb_categorical = OptimalBinning(
        name=var,
        dtype="categorical",
        min_n_bins=min_n_bins,
        special_codes=special_list,
        cat_cutoff=cat_cutoff,
        min_bin_size=min_bin_size,
        max_n_bins=max_n_bins
    )
    return optb_categorical.fit(x, y)


def get_iv_data(object_dict):
    """Generate Information Value (IV) data frames
    
    Parameters:
    -----------
    object_dict : dict
        Dictionary containing variable names and binning objects
        
    Returns:
    --------
    pd.DataFrame
        Concatenated Information Value data frames
    """
    iv_data_frames = []

    for variable, binning_object in object_dict.items():
        try:
            iv_df = binning_object.binning_table.build().iloc[:-1, :-1]
            iv_df['Variable'] = variable
            iv_df = iv_df.rename(columns={"IV": "IV(Bins)"})
            iv_df['IV'] = iv_df['IV(Bins)'].sum()
            iv_df = iv_df[['Variable', 'Bin', 'Count', 'Count (%)', 'Non-event', 
                           'Event', 'Event rate', 'WoE', 'IV(Bins)', 'IV']]
            iv_data_frames.append(iv_df)
        except Exception as e:
            print(f"Error processing {variable}: {str(e)}")

    if not iv_data_frames:
        return pd.DataFrame()

    final_iv_df = pd.concat(iv_data_frames, axis=0, ignore_index=True)
    final_iv_df['Bin'] = final_iv_df['Bin'].astype(str)
    return final_iv_df


def transformer_numeric(optb_fit_objects, data, variable, metric='bins'):
    """Transform a numerical variable
    
    Parameters:
    -----------
    optb_fit_objects : dict
        Dictionary containing OptimalBinning objects
    data : pd.DataFrame
        Input data
    variable : str
        Variable to transform
    metric : str
        Transformation metric
        
    Returns:
    --------
    pd.Series
        Transformed variable
    """
    try:
        transformer_numeric = optb_fit_objects[variable]
        tx = transformer_numeric.binning_table.build()
        metric_missing = tx.loc[tx['Bin'].apply(lambda x: 'Missing' in x), 'WoE'].tolist()[0]

        x_binned_numeric = transformer_numeric.transform(
            data[variable].values, 
            metric=metric, 
            metric_missing=metric_missing
        )
        return x_binned_numeric
    except Exception as e:
        print(f"Error in transformer_numeric for variable '{variable}': {str(e)}")
        return None


def transformer_categorical(optb_fit_objects, data, variable, metric='bins'):
    """Transform a categorical variable
    
    Parameters:
    -----------
    optb_fit_objects : dict
        Dictionary containing OptimalBinning objects
    data : pd.DataFrame
        Input data
    variable : str
        Variable to transform
    metric : str
        Transformation metric
        
    Returns:
    --------
    pd.Series
        Transformed variable
    """
    try:
        transformer_categorical = optb_fit_objects[variable]
        tx = transformer_categorical.binning_table.build()
        metric_missing = tx.loc[tx['Bin'].apply(lambda x: 'Missing' in x), 'WoE'].tolist()[0]
        
        x_binned_categorical = transformer_categorical.transform(
            data[variable].values, 
            metric=metric,
            metric_missing=metric_missing
        )
        return x_binned_categorical
    except Exception as e:
        print(f"Error in transformer_categorical for variable '{variable}': {str(e)}")
        return None


def transformer_other_pandas(df, optb_fit_objects, uid, target, metric='bins'):
    """Transform a pandas DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    optb_fit_objects : dict
        Dictionary containing OptimalBinning objects
    uid : str
        Unique identifier column
    target : str
        Target column
    metric : str
        Transformation metric
        
    Returns:
    --------
    pd.DataFrame
        Transformed DataFrame
    """
    try:
        X = df.drop(columns=[uid, target])
        y = df[target]

        numerical_columns = X.select_dtypes(include=['number']).columns
        categorical_columns = X.select_dtypes(exclude=['number']).columns
        part_df_transformed = df[[uid, target]]

        # Transform numerical columns
        if not numerical_columns.empty:
            for variable in numerical_columns:
                try:
                    x_transformed_numeric = transformer_numeric(
                        optb_fit_objects, X, variable, metric=metric
                    )
                    if x_transformed_numeric is not None:
                        part_df_transformed = pd.concat(
                            [part_df_transformed, pd.Series(x_transformed_numeric, name=f"{metric}_{variable}")], 
                            axis=1
                        )
                except Exception as e_numeric:
                    print(f"Error transforming numeric variable '{variable}': {str(e_numeric)}")

        # Transform categorical columns
        if not categorical_columns.empty:
            for variable in categorical_columns:
                try:
                    x_transformed_categorical = transformer_categorical(
                        optb_fit_objects, X, variable, metric=metric
                    )
                    if x_transformed_categorical is not None:
                        part_df_transformed = pd.concat(
                            [part_df_transformed, pd.Series(x_transformed_categorical, name=f"{metric}_{variable}")], 
                            axis=1
                        )
                except Exception as e_categorical:
                    print(f"Error transforming categorical variable '{variable}': {str(e_categorical)}")

        return part_df_transformed

    except Exception as e_main:
        print(f"Error in transformer_other_pandas: {str(e_main)}")
        return None


def rf_pimp_feature_importance(data, id_col, target, selected_features, factor=None, model_params=None):
    """Calculate feature importance using Random Forest
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input DataFrame
    id_col : str
        ID column name
    target : str
        Target column name
    selected_features : list
        List of features to consider
    factor : float
        Importance factor
    model_params : dict
        Model parameters
        
    Returns:
    --------
    tuple
        Result DataFrame and list of selected features
    """
    try:
        # Initialize model
        base_model = LGBMClassifier()
        if model_params is None:
            model_params = {
                'n_estimators': 100,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1,
            }
            
        # Prepare data
        df = data.copy()
        X = df[selected_features].copy()  # Create explicit copy to avoid SettingWithCopyWarning
        y = df[target]

        # Handle missing values
        numerical_columns = X.select_dtypes(include=['number']).columns
        categorical_columns = X.select_dtypes(exclude=['number']).columns
        label_encoder = LabelEncoder()

        # Use proper indexing to avoid warnings
        for column in numerical_columns:
            X.loc[:, column] = X[column].fillna(-999)
            
        for column in categorical_columns:
            X.loc[:, column] = X[column].fillna("-999")
            X.loc[:, column] = label_encoder.fit_transform(X[column])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        
        # Temporarily suppress warnings during model training
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Train model
            rf_model = base_model
            rf_model.set_params(**model_params)
            rf_model.fit(X_train, y_train)
            
            # Evaluate model
            rf_model.score(X_test, y_test)
            
            # Calculate permutation importance
            scoring = ['r2', 'neg_mean_absolute_percentage_error', 'neg_mean_squared_error']
            r_multi = permutation_importance(
                rf_model, X_test, y_test, 
                n_repeats=10, random_state=0, 
                n_jobs=-1, scoring=scoring
            )
        
        # Process results
        results_list = []
        feature_names = X.columns.tolist()
        
        for metric in r_multi:
            r = r_multi[metric]
            for i in r.importances_mean.argsort()[::-1]:
                if r.importances_mean[i] - factor * r.importances_std[i] > 0:
                    results_list.append({
                        'Metric': metric,
                        'Feature': feature_names[i],
                        'Importance Mean': r.importances_mean[i],
                        'Importance Std': r.importances_std[i]
                    })
        
        # Create results DataFrame
        rf_pimp_result_df = pd.DataFrame(results_list)
        cols_after_rf_pimp = rf_pimp_result_df['Feature'].unique().tolist()
        
        return rf_pimp_result_df, cols_after_rf_pimp
        
    except Exception as e:
        print(f"Error in rf_pimp_feature_importance: {str(e)}")
        return None, None


def shap_rfecv(df, features, target, min_features=10):
    """Perform SHAP-based recursive feature elimination
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    features : list
        List of features to consider
    target : str
        Target column name
    min_features : int
        Minimum number of features to select
        
    Returns:
    --------
    tuple
        List of selected features and feature elimination report
    """
    if len(features) > min_features:
        train = df
        
        # Initialize model
        clf = LGBMClassifier(max_depth=5, class_weight='balanced', verbose=-1)
        param_grid = {'n_estimators': [5, 7, 10], 'num_leaves': [3, 5, 7, 10]}
        
        from sklearn.model_selection import RandomizedSearchCV
        search = RandomizedSearchCV(clf, param_grid)
        
        # Prepare data
        X = train[features]
        y = train[target].astype('int')
        
        # Initialize SHAP-based feature elimination
        shap_elimination = ShapRFECV(
            model=search,
            min_features_to_select=min_features,
            step=0.1, 
            cv=5, 
            scoring='roc_auc', 
            n_jobs=-1,
            verbose=0
        )
        
        # Fit and compute elimination
        report = shap_elimination.fit_compute(X, y, check_additivity=False)
        
        # Find optimal number of features
        t1_score = report['train_metric_mean'].max()
        report['BestValue'] = (report['train_metric_mean'] * 100 / t1_score)
        
        num_of_features = int(
            report.sort_values(
                by=['BestValue', 'num_features'],
                ascending=[False, True]
            )['num_features'].reset_index(drop=True)[0]
        )
        
        # Get selected features
        selected_features = list(shap_elimination.get_reduced_features_set(num_features=num_of_features))
    else:
        selected_features = features
        report = 'Not available'
        
    return selected_features, report


def ml_data_imputer(data, numerical_cols, cat_cols, target, test_size=0.3):
    """Prepare data for machine learning by imputing missing values and encoding categorical variables
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input DataFrame
    numerical_cols : list
        List of numerical columns
    cat_cols : list
        List of categorical columns
    target : str
        Target column name
    test_size : float
        Test size for train-test split
        
    Returns:
    --------
    tuple
        Training and test DataFrames
    """
    # One-hot encode categorical variables
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    
    # Convert categorical columns to category type
    for col in cat_cols:
        data[col] = data[col].astype('category')
    
    # Apply one-hot encoding
    if cat_cols:
        df_encoded = pd.DataFrame(
            encoder.fit_transform(data[cat_cols]), 
            columns=encoder.get_feature_names_out(cat_cols)
        )
        
        # Concatenate encoded columns with original DataFrame
        data_encoded = pd.concat([data, df_encoded], axis=1)
        
        # Drop original categorical columns
        data_encoded = data_encoded.drop(cat_cols, axis=1)
    else:
        data_encoded = data.copy()
    
    # Split data into train and test sets
    train_ml, test_ml = train_test_split(data_encoded, test_size=test_size)
    
    # Scale numerical features
    if numerical_cols:
        scaler = MinMaxScaler()
        scaler.fit(train_ml[numerical_cols])
        train_ml[numerical_cols] = scaler.transform(train_ml[numerical_cols])
        test_ml[numerical_cols] = scaler.transform(test_ml[numerical_cols])
    
    return train_ml, test_ml 