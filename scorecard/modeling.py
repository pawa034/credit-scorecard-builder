import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, log_loss, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import statsmodels.api as sm
import pickle
import os


def get_models(seed=None):
    """Generate a library of base learners
    
    Parameters:
    -----------
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary of classifier models
    """
    SEED = seed
    models = {
        'knn': KNeighborsClassifier(n_neighbors=3),
        'naive bayes': GaussianNB(),
        'mlp-nn': MLPClassifier((80, 10), early_stopping=False, random_state=SEED),
        'random forest': RandomForestClassifier(n_estimators=10, max_features=3, random_state=SEED),
        'gbm': GradientBoostingClassifier(n_estimators=100, random_state=SEED),
        'logistic': LogisticRegression(C=100, random_state=SEED),
    }
    
    return models


def train_predict(models, x_dev, x_oot, y_dev, y_oot):
    """Train multiple classifiers and make predictions
    
    Parameters:
    -----------
    models : dict
        Dictionary of classifier models
    x_dev : pd.DataFrame
        Features of development dataset
    x_oot : pd.DataFrame
        Features of out-of-time dataset
    y_dev : pd.Series
        Target of development dataset
    y_oot : pd.Series
        Target of out-of-time dataset
        
    Returns:
    --------
    pd.DataFrame
        Predictions from each model
    """
    P = np.zeros((y_oot.shape[0], len(models)))
    P = pd.DataFrame(P)

    print("Fitting models".center(100, '-'))
    cols = list()
    
    for i, (name, m) in enumerate(models.items()):
        print(f"{name}...")
        m.fit(x_dev, y_dev)
        P.iloc[:, i] = m.predict_proba(x_oot)[:, 1]
        cols.append(name)

    P.columns = cols
    return P


def score_models(P, y_oot):
    """Score models based on predictions
    
    Parameters:
    -----------
    P : pd.DataFrame
        Predictions from multiple models
    y_oot : pd.Series
        True target values
        
    Returns:
    --------
    pd.DataFrame
        Model names and performance scores
    """
    print('\n')
    print("Scoring models".center(100, '-'))
    
    model_name = "Model_Name"
    log_loss_ = "logLoss"
    print(f"%-26s  %-26s" % (model_name, log_loss_))
    
    models = []
    scores = []
    
    for m in P.columns:
        score = roc_auc_score(y_oot, P.loc[:, m])
        logLoss = log_loss(y_oot, P.loc[:, m])
        print(f"%-26s: %.3f" % (m, logLoss))
        models.append(m)
        scores.append(logLoss)
    
    print("COMPLETED...\n")
    print("".center(100, '-'))
    
    return pd.DataFrame({'ModelName': models, 'Score': scores})


def ml_summary(data, oot, binnable_feats=None, target=None, models_func=None):
    """Generate ML model summary
    
    Parameters:
    -----------
    data : pd.DataFrame
        Development dataset
    oot : pd.DataFrame
        Out-of-time dataset
    binnable_feats : list
        Features to use for modeling
    target : str
        Target column name
    models_func : function
        Function that returns a dictionary of models
        
    Returns:
    --------
    tuple
        Feature importances, best model, model name, dev data with predictions, oot data with predictions
    """
    if models_func is None:
        models_func = get_models
        
    # Add prediction column
    data['prob_ml'] = ''
    oot['prob_ml'] = ''
    
    # Prepare data
    x_dev = data[binnable_feats]
    x_oot = oot[binnable_feats]
    y_dev = data[target]
    y_oot = oot[target]
    
    # Handle missing values
    x_dev = x_dev.fillna(0)
    x_oot = x_oot.fillna(0)
    y_dev = y_dev.fillna(0)
    y_oot = y_oot.fillna(0)
    
    # Get and train models
    models = models_func(123)
    P = train_predict(models, x_dev, x_oot, y_dev, y_oot)
    
    # Score models
    score = score_models(P, y_oot)
    
    # Plot ROC curves
    from scorecard.utils import plot_roc_curve
    plot_roc_curve(y_oot, P.values, P.mean(axis=1), list(P.columns), "ensemble")
    
    # Find best model
    min_val = score['Score'].min()
    best_model = score[score['Score'] == min_val]['ModelName'].values[0]
    model = models[best_model].fit(x_dev, y_dev)
    
    print('\n')
    print(f'Best Model fitted on Raw Data is : {best_model}')
    print('\n')
    
    # Add predictions to data
    data['prob_ml'] = 1 - model.predict_proba(x_dev)
    oot['prob_ml'] = 1 - model.predict_proba(x_oot)
    df_dev = data.copy()
    df_oot = oot.copy()
    
    # Extract feature importances if applicable
    feature_importances = None
    if best_model in ['gbm', 'random forest']:
        feature_importances = pd.DataFrame(
            model.feature_importances_,
            index=x_dev.columns,
            columns=['importance']
        ).sort_values('importance', ascending=False)
        print(f'Feature ranking for > {best_model}')
        print('\n')
        print(feature_importances.to_string())
    else:
        feature_importances = pd.Series(binnable_feats)
        print(feature_importances.to_string())
    
    return feature_importances, model, best_model, df_dev, df_oot


def gen_ks_calculator(data=None, target=None, prob=None):
    """Generate KS table for scored population
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with scored data
    target : str
        Target column name
    prob : str
        Probability column name
        
    Returns:
    --------
    tuple
        KS statistic, decile with max KS, monotonicity flag, KS table
    """
    try:
        data['target0'] = 1 - data[target]
        data['bucket'] = pd.qcut(data[prob], 10)
        
        grouped = data.groupby('bucket', as_index=False)
        kstable = pd.DataFrame()
        
        # Calculate KS statistics
        kstable['min_prob'] = grouped.min()[prob]
        kstable['max_prob'] = grouped.max()[prob]
        kstable['events'] = grouped.sum()[target]
        kstable['nonevents'] = grouped.sum()['target0']
        kstable['count'] = kstable['events'] + kstable['nonevents']
        
        # Sort by probability
        kstable = kstable.sort_values(by="min_prob", ascending=False).reset_index(drop=True)
        
        # Calculate rates
        kstable['event_rate'] = (kstable.events / kstable['count']).apply('{0:.2%}'.format)
        kstable['badrate'] = (kstable.events / kstable['count'])
        kstable['nonevent_rate'] = (kstable.nonevents / data['target0'].sum()).apply('{0:.2%}'.format)
        
        # Calculate cumulative rates
        kstable['cum_eventrate'] = (kstable.events / data[target].sum()).cumsum()
        kstable['cum_noneventrate'] = (kstable.nonevents / data['target0'].sum()).cumsum()
        
        # Calculate KS
        kstable['KS'] = np.round(kstable['cum_eventrate'] - kstable['cum_noneventrate'], 3) * 100
        
        # Format outputs
        kstable['cum_eventrate'] = kstable['cum_eventrate'].apply('{0:.2%}'.format)
        kstable['cum_noneventrate'] = kstable['cum_noneventrate'].apply('{0:.2%}'.format)
        kstable.index = range(1, 11)
        kstable.index.rename('Decile', inplace=True)
        
        # Check monotonicity
        monotonicity_flag = kstable['badrate'].is_monotonic_decreasing
        
        # Extract max KS and its decile
        max_ks = max(kstable['KS'])
        ks_decile = str(kstable.index[kstable['KS'] == max(kstable['KS'])][0])
        
        return max_ks, ks_decile, monotonicity_flag, kstable
        
    except Exception as e:
        print(f'KS calculation failed: {str(e)}')
        print('Probabilities achieved not unique, Model Conversion Failed')
        return 'FAILED', 'FAILED', 'FAILED', 'FAILED'


def ks_calculator(data=None, target=None, prob=None):
    """Calculate KS statistics
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data with predictions
    target : str
        Target column name
    prob : str
        Probability column name
        
    Returns:
    --------
    tuple
        Max KS, KS decile, monotonicity flag, KS table
    """
    try:
        data['target0'] = 1 - data[target]
        data['bucket'] = pd.qcut(data[prob], 10)
        
        grouped = data.groupby('bucket', as_index=False)
        kstable = pd.DataFrame()
        
        # Calculate KS statistics
        kstable['min_prob'] = grouped.min()[prob]
        kstable['max_prob'] = grouped.max()[prob]
        kstable['events'] = grouped.sum()[target]
        kstable['nonevents'] = grouped.sum()['target0']
        kstable['count'] = kstable['events'] + kstable['nonevents']
        
        # Sort by probability
        kstable = kstable.sort_values(by="min_prob", ascending=False).reset_index(drop=True)
        
        # Calculate rates
        kstable['event_rate'] = (kstable.events / kstable['count']).apply('{0:.2%}'.format)
        kstable['badrate'] = (kstable.events / kstable['count'])
        kstable['nonevent_rate'] = (kstable.nonevents / data['target0'].sum()).apply('{0:.2%}'.format)
        
        # Calculate cumulative rates
        kstable['cum_eventrate'] = (kstable.events / data[target].sum()).cumsum()
        kstable['cum_noneventrate'] = (kstable.nonevents / data['target0'].sum()).cumsum()
        
        # Calculate KS
        kstable['KS'] = np.round(kstable['cum_eventrate'] - kstable['cum_noneventrate'], 3) * 100
        
        # Format outputs
        kstable['cum_eventrate'] = kstable['cum_eventrate'].apply('{0:.2%}'.format)
        kstable['cum_noneventrate'] = kstable['cum_noneventrate'].apply('{0:.2%}'.format)
        kstable.index = range(1, 11)
        kstable.index.rename('Decile', inplace=True)
        
        # Check monotonicity
        monotonicity_flag = kstable['badrate'].is_monotonic_decreasing
        
        # Extract max KS and its decile
        max_ks = max(kstable['KS'])
        ks_decile = str(kstable.index[kstable['KS'] == max(kstable['KS'])][0])
        
        return max_ks, ks_decile, monotonicity_flag, kstable
        
    except Exception as e:
        print(f'KS calculation failed: {str(e)}')
        print('Probabilities achieved not unique, Model Conversion Failed')
        return 'FAILED', 'FAILED', 'FAILED', 'FAILED'


def gen_ks_calculator_groups(data=None, target=None, prob=None, groups=None):
    """Generate KS table with custom grouping
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data with predictions
    target : str
        Target column name
    prob : str
        Probability column name
    groups : int
        Number of groups for binning
        
    Returns:
    --------
    pd.DataFrame
        KS table
    """
    data['target0'] = 1 - data[target]
    data['bucket'] = pd.qcut(data[prob], groups, labels=False, duplicates='drop')
    
    grouped = data.groupby('bucket', as_index=False)
    kstable = pd.DataFrame()
    
    # Calculate KS statistics
    kstable['min_prob'] = grouped.min()[prob]
    kstable['max_prob'] = grouped.max()[prob]
    kstable['events'] = grouped.sum()[target]
    kstable['nonevents'] = grouped.sum()['target0']
    kstable['count'] = kstable['events'] + kstable['nonevents']
    
    # Sort by probability
    kstable = kstable.sort_values(by="min_prob", ascending=False).reset_index(drop=True)
    
    # Calculate rates
    kstable['event_rate'] = (kstable.events / kstable['count']).apply('{0:.2%}'.format)
    kstable['badrate'] = (kstable.events / kstable['count'])
    kstable['nonevent_rate'] = (kstable.nonevents / data['target0'].sum()).apply('{0:.2%}'.format)
    
    # Calculate cumulative rates
    kstable['cum_eventrate'] = (kstable.events / data[target].sum()).cumsum()
    kstable['cum_noneventrate'] = (kstable.nonevents / data['target0'].sum()).cumsum()
    
    # Calculate KS
    kstable['KS'] = np.round(kstable['cum_eventrate'] - kstable['cum_noneventrate'], 3) * 100
    
    # Format outputs
    kstable['cum_eventrate'] = kstable['cum_eventrate'].apply('{0:.2%}'.format)
    kstable['cum_noneventrate'] = kstable['cum_noneventrate'].apply('{0:.2%}'.format)
    kstable = kstable.reset_index()
    kstable.index.rename('Decile', inplace=True)
    
    return kstable


def modeling(no_of_iter=10, varlist=None, data=None, target=None):
    """Perform logistic regression
    
    Parameters:
    -----------
    no_of_iter : int
        Number of iterations
    varlist : list
        List of predictor variables
    data : pd.DataFrame
        Input data
    target : str
        Target column name
        
    Returns:
    --------
    statsmodels.genmod.generalized_linear_model.GLMResultsWrapper
        Logistic regression results
    """
    if varlist is None or data is None or target is None:
        raise ValueError("varlist, data, and target are required inputs.")

    X_dev = data[varlist]
    Y_dev = data[target]
    
    logit_model = sm.Logit(Y_dev, X_dev)
    result = logit_model.fit(maxiter=no_of_iter, disp=False)
    
    return result


def modeling2(n=10, varlist=None, data=None, target=None):
    """Alternative logistic regression implementation
    
    Parameters:
    -----------
    n : int
        Maximum number of iterations
    varlist : list
        List of predictor variables
    data : pd.DataFrame
        Input data
    target : str
        Target column name
        
    Returns:
    --------
    statsmodels.genmod.generalized_linear_model.GLMResultsWrapper
        Logistic regression results
    """
    X_dev = data[varlist]
    Y_dev = data[target]
    
    X_dev = sm.add_constant(X_dev)
    logit_model = sm.Logit(Y_dev, X_dev)
    result = logit_model.fit(disp=False)
    
    return result


def predict_method(data, vars1, result):
    """Add predicted probabilities to data
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    vars1 : list
        List of predictor variables
    result : statsmodels.genmod.generalized_linear_model.GLMResultsWrapper
        Fitted model
        
    Returns:
    --------
    pd.DataFrame
        Data with predictions
    """
    x = data[vars1]
    x = sm.add_constant(x)
    data['prob'] = result.predict(x)
    
    return data


def p_val_calc(result):
    """Extract significant variables based on p-values
    
    Parameters:
    -----------
    result : statsmodels.genmod.generalized_linear_model.GLMResultsWrapper
        Fitted model
        
    Returns:
    --------
    list
        List of significant variables
    """
    cov = result.cov_params()
    sum1 = pd.DataFrame({
        "Z_SCORE": result.params / np.sqrt(np.diag(cov)),
        'pvals': result.pvalues,
        'p_values': (result.pvalues < 0.05)
    }).reset_index()
    
    sum1['z_abs'] = abs(sum1['Z_SCORE'])
    sum2 = pd.DataFrame(sum1.sort_values(by='pvals', ascending=True).reset_index().drop('level_0', axis=1))
    sum2 = sum2.rename(columns={'index': 'Vars'})
    
    significant_vars = list(sum2[sum2["p_values"] == True]["Vars"])
    
    return [col for col in significant_vars if col != 'const']


def prob_to_score(data=None, pdo=None, constant=None, prob_column=None):
    """Convert probabilities to scores
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data with probabilities
    pdo : int
        Points to double odds
    constant : int
        Base score
    prob_column : str
        Probability column name
        
    Returns:
    --------
    pd.DataFrame
        Data with scores
    """
    prob = prob_column
    df = data
    
    # Calculate odds and transform to score
    data['prob0'] = 1 - data[prob]
    df['a'] = np.log(51200)
    df['f'] = np.log(df['prob0'] / df[prob])
    df['c'] = np.log(0.048828125)
    df['min'] = df[['a', 'f']].min(axis=1)
    df['logOdds'] = df[['min', 'c']].max(axis=1)
    df['factor'] = pdo / np.log(2)
    df['constant'] = constant
    df['offset'] = df['constant'] - (df['factor'] * np.log(50))
    df['score'] = (round(df['offset'] + (df['factor'] * df['logOdds']))).astype(int)
    
    # Clean up temporary columns
    cols_to_del = ['a', 'f', 'c', 'min', 'logOdds', 'factor', 'constant', 'offset']
    df = df.drop(cols_to_del, axis=1)
    
    print('Score Density Plot for the Current data')
    ax = df['score'].plot.kde()
    
    return df


def log_loss_calc(model, X, y):
    """Calculate log loss for model evaluation
    
    Parameters:
    -----------
    model : object
        Fitted model
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
        
    Returns:
    --------
    float
        Log loss value
    """
    pred = model.predict_proba(X)
    logloss = log_loss(y, pred)
    
    return logloss 