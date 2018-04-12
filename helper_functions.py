## Functions supporting the Project notebook

import warnings
import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import random as rnd
import gspread
import sklearn.preprocessing
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse
from operator import add
import itertools

# Suppress annoying red box warnings
warnings.filterwarnings('ignore')

# Read in data from Google sheet
def import_sheet(name):
    """Use Google Sheets API to read in latest verison of data file
    
    Keyword arguments:
    name -- name of Google sheet (string)
    """
    
    # Use credentials to create a client to interact with the Google Drive API
    scope = ['https://spreadsheets.google.com/feeds']
    creds = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json', scope)
    client = gspread.authorize(creds)

    # Read the spreadsheet and convert to a Pandas dataframe
    sheet = client.open(name).sheet1
    df = pd.DataFrame(sheet.get_all_records(empty2zero=True)) # will be a column name alphabetized dataframe
    
    # Convert the dataframe to numeric
    df.astype('float64')
    return df


# Flip dataframe so most recent times are first
def flip_df(orig_df):

    df = orig_df.copy()
    df = df.sort_index(ascending=False)
    df.reset_index(drop=True, inplace=True)
    return df


# Standardiz feature ranges to do feature construction
def standardize(orig_df, feature_range=[2,4], comp_var='Price'):
    """
    Scale all of a dataframe's columns linearly to some range
    """

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=feature_range)
    std_array = scaler.fit_transform(orig_df)
    std_df = pd.DataFrame(data=std_array, columns=list(orig_df))

    min_cv = orig_df[comp_var].min()
    max_cv = orig_df[comp_var].max()

    def rescaler(y):

        return (y-feature_range[0])*(max_cv-min_cv)/(feature_range[1]-feature_range[0])+min_cv
    
    return std_df, rescaler


# Plot all datframe signals
def signal_plot(orig_df):
    """Create an overlaid time series plot of all features in a dataframe.
    
    Keyword arguments:
    """

    df = orig_df.copy()
    df, not_used = standardize(df, feature_range=[2,4])
    df['time'] = df.index
    df['unit'] = np.zeros((len(df),1))
    df_melt = pd.melt(df,id_vars=['time','unit'],value_vars=list(orig_df))
    plt.figure(figsize=(60,14))
    tsplot = sns.tsplot(data=df_melt, time='time', value='value',condition='variable',unit='unit')
    plt.show()


# Regular heatmap
def reg_heat_map(orig_df, include_mask=False, include_annot=False):
    """Create a heat map of Pearson correlation coefficients between all features.
    
    Keyword arguments:
    orig_df -- a pandas dataframe
    include_mask -- True or False, whether or not to hide the upper triangle
    include_annot -- True or False, whether or not to write numbers in the boxes
    """

    df = orig_df.copy()
    num_cols = df.shape[1]

    # Compute correlation coefficients
    corr = df.corr()
    
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    if include_mask:
        mask[np.triu_indices_from(mask)] = True
    
    # Set up the matplotlib figure
    plt.subplots(figsize=(num_cols/3,num_cols/3))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(np.round(corr,2), mask=mask, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=0, annot=include_annot, annot_kws={"size": 7}, cbar_kws={"shrink":1})
    plt.show()
    
 
# Pearson coeff
def pearson(sig1,sig2):
    """Calculate the Pearson Correlation Coefficient of two signals (two numpy vectors)"""
    # print(sig1)
    # print(sig2)
    return np.cov(np.array([sig1,sig2]))[0,1]/(np.std(sig1)*np.std(sig2))


# Time heatmap
def time_heat_map(orig_df, comp_var='Price', months=60, include_annot=False):
    """Create a heatmap of Pearson correlation coefficients where all features are 
    compared to one feature at different time delays. Rows are sorted by best maximum observed
    correlation with price.

    Keyword arguments:
    orig_df: pandas dataframe
    months: number of months to look back in time
    comp_var: name of feature to compare to all others (string, e.g. 'Price')
    include_annot: True or False, whether or not to write numbers in the boxes
    """

    df = orig_df.copy()

    num_cols = df.shape[1] # Number of dataframe columns

    # Create an empty correlation matrix. Rows will be features and columns will be time delays.
    time_corr = np.zeros((num_cols,months+1))
    #feature_names = list(df.drop('Price',axis=1))
    feature_names = list(df)

    for i in range(num_cols): # For every feature
        for j in range(months+1):    # For every month delay, from 0 up to and including n
            # Truncate beginning of price signal
            comp_var_sig = df[comp_var][j:len(df[comp_var])+1]
            # Truncate end of feature signal
            feature_name = feature_names[i]
            #print(feature_name)
            feature_sig = df[feature_name][0:len(df[feature_name])-j]
            time_corr[i,j] = np.round(pearson(comp_var_sig,feature_sig),2)

    # Set up the matplotlib figure
    plt.subplots(figsize=(months/3,num_cols/3))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Create dataframe from matrix of correlations
    time_corr_df = pd.DataFrame(data=np.transpose(time_corr),columns = feature_names)
    time_corr_df = time_corr_df.reindex_axis(time_corr_df.abs().max().sort_values().index, axis=1)
    #time_corr_df = time_corr_df.reindex_axis(time_corr_df.mean().sort_values().index, axis=1)

    # Generate heat map
    sns.heatmap(time_corr_df.transpose(), cmap=cmap, vmax=1, center=0,
                square=False, linewidths=0, annot=include_annot, annot_kws={"size": 7}, cbar_kws={"shrink":1})
    plt.show()


def good_corr(orig_df, feature, comp_var='Price', corr_cut=0.8, look_back=7):

    good = False
    pearson_coeffs = [];
    for i in range(look_back+1):
        comp_var_sig = orig_df[comp_var][i:len(orig_df[comp_var])+1]
        feature_sig = feature[0:len(feature)-i]
        pearson_coeffs.append( pearson(comp_var_sig,feature_sig) )
    good = np.max(np.abs(pearson_coeffs)) >= corr_cut

    return good


def cumtrapz_ext(signal):

    return np.hstack((0,sp.integrate.cumtrapz(signal)))


def diff_ext(signal):

    return np.hstack((0,np.diff(signal)))


# Construct new features with function transformations
def new_features_with_funcs(orig_df, funcs, func_names, comp_var='Price', filter=True, corr_cut=0, look_back=7):
    """Create new feature columns out of old feature columns by applying transforming functions
     
    Keyword arguments:
    orig_df: pandas dataframe
    funcs: list of functions to apply to data
	func_names: list of names of functions to apply to data (for naming new columns)
    comp_var: feature of interest, to compare all other features to
    """

    df = orig_df.copy()
    col_names = list(df)
    # col_names.remove(comp_var)

    new_feats, thrown_out = 0, 0

    for col_name in col_names:
        for i in range(len(funcs)):
            new_feature = funcs[i](df[col_name])
            if filter == True:
                if good_corr(df, new_feature, comp_var=comp_var, corr_cut=corr_cut, look_back=look_back):
                    df[col_name+func_names[i]] = funcs[i](df[col_name])
                    new_feats +=1
                else:
                    thrown_out += 1
            else:
                df[col_name+func_names[i]] = funcs[i](df[col_name])
                new_feats +=1
    print(str(new_feats)+' net new features created with functions. '+str(thrown_out)+' new features were thrown out due to poor correlation with price at all month offsets from 0 to '+ str(look_back))
    return df


# Construct new features with combination
def new_features_with_combs(orig_df, combiners=['*','/'], comp_var='Price', filter=True, corr_cut=0.8, look_back=7):
    """Create new feature columns out of old feature columns by combining features with multiplication and division.
    If a new feature is poorly correlated with price (incl any time shift up to look_back months), throw it out.
     
    Keyword arguments:
    orig_df: pandas dataframe
    comp_var: name of feature to try out correlations of new features
    corr_cut: Pearson correlation with comp_var needs to be above this value to qualify
    look_back: check Pearson correlation with comp_var offsetting up to this many months in the past
    """

    df = orig_df.copy()
    col_names = list(df)
    # col_names.remove(comp_var)

    new_feats = {}
    combinations = [comb for comb in itertools.combinations(col_names, 2)]
    num_new = len(combinations)*3
    for col1, col2 in combinations:
        if col1+'*'+col2 not in col_names and col1+'/'+col2 not in col_names and col2+'/'+col1 not in col_names:
            for combiner in combiners:
                if combiner == '*':
                    new_feats[col1+'*'+col2] = df[col1].multiply(df[col2])
                if combiner == '/':
                    new_feats[col1+'/'+col2] = df[col1].divide(df[col2])
                    new_feats[col2+'/'+col1] = df[col2].divide(df[col1])
    if filter == True:
        i = 0
        for key in list(new_feats.keys()):
            i+=1
            print('Testing new feature {0} of {1}'.format(i,num_new), end='         \r')
            if not good_corr(df, new_feats[key], comp_var=comp_var, corr_cut=corr_cut, look_back=look_back):
                new_feats.pop(key) 
    new_feats_df = pd.DataFrame(data=new_feats)
    df = pd.concat([df, new_feats_df], axis=1)
    print(str(len(new_feats))+' net new features created with combinations. '+str(num_new-len(new_feats))+' new features were thrown out due to poor correlation with '+comp_var)
    return(df)         


# Delete features that are poorly correlated with comp_var at several offsets
def trim_features(orig_df, comp_var='Price', corr_cut=0.7, look_back=7):

    df = orig_df.copy()
    col_names = list(df)
    col_names.remove(comp_var)

    thrown_out = 0

    for col_name in col_names:
        feature = df[col_name]
        if not good_corr(df, feature, comp_var=comp_var, corr_cut=corr_cut, look_back=look_back):
            df = df.drop((col_name), axis=1)
            thrown_out += 1
    print(str(thrown_out)+' features thrown out due to poor correlation with price at all month offsets from 0 to '+ str(look_back))
    return df


# Delete features that are highly correlated to other features
def keep_distinct_features(orig_df, comp_var='Price', corr_cut=0.7, look_back=7):
    """Remove features that are highly correlated to one another at all points in time.

    For each feature in the dataframe, check Pearson correlation with all other features with no time offset.
    If the correlation is high, continue checking with look_back time offsets. If all of these correlations are high, 
    add the feature to a list for correlation comparison with comp_var. Keep only the best of that list.

    Keyword arguments:
    orig_df: pandas dataframe
    corr_cut: Pearson correlation above this value means the features are highly correlated.
    look_back: check Pearson correlation with comp_var up to this many months offset to pick best
    """

    df = orig_df.copy()
    col_names = list(df)
    col_names.remove(comp_var)
    num = len(col_names)
    n, m, thrown_out, nans = 0, 0, 0, 0

    for col_name1 in col_names:
        n += 1
        m =0
        num = len(list(df))
        # print('Checking {0} of {1}'.format(n, num), end='         \r' )
        if col_name1 not in list(df):
            continue
        bundle = [col_name1]
        for col_name2 in col_names:
            m += 1
            print('Checking {0} with {1} of {2} total '.format(n, m, num), end='         \r' )
            if col_name2 not in list(df):
                continue
            if col_name1 != col_name2:
                # print(col_name1)
                # print(col_name2)
                corr = [pearson(df[col_name1],df[col_name2])]
                if np.isnan(corr[0]):
                    nans += 1;
                    print(col_name1)
                    print(col_name2)
                if abs(corr[0]) >= corr_cut:
                    bundle.append(col_name2)
        # Keep signal with highest correlation to 'comp_var', looking back 'look_back' months
        if len(bundle) > 1 and look_back >= 1:
            comp_var_corr = []
            for col_name in bundle:
                best_corr = 0
                for i in range(1,look_back):
                    comp_var_sig = df[comp_var][i:len(df[comp_var])+1]
                    feat_sig = df[col_name][0:len(df[col_name])-i]
                    corr = pearson(comp_var_sig,feat_sig)
                    if abs(corr) >= best_corr:
                        best_corr = corr
                comp_var_corr.append(best_corr)
            max_index = comp_var_corr.index(max(comp_var_corr))
            #print('Matching:')
            #print(bundle)
            bundle.pop(max_index)
            df = df.drop(labels=bundle, axis=1)
            thrown_out += len(bundle)
            #print('Removed:')
            #print(bundle)

    print(str(thrown_out)+' features thrown out due to correlation with other features. '+str(nans)+' NaNs resulted from Pearson calculations')
    return df


# Manipulate dataframe to make time offsets real features (columns) for training
def slide_df(orig_df, offset, look_back, purpose='train', comp_var='Price'):

    orig_df_reindexed = orig_df.reset_index(drop=True)
    i = offset
    new_df_array = np.ndarray((1,1)) # new dataframe to learn on, composed of look_back slid rows of a slice starting i months in the past
    orig_num_rows = orig_df_reindexed.shape[0]
    if purpose == 'train':
        new_num_rows = orig_num_rows + 1 - i - look_back # how many rows the dataframe will have (maximum available... -i for prediction offset, -look_back for equal sliding of rows
    elif purpose == 'test':
        new_num_rows = 1
    new_df_colnames = [comp_var]
    for j in range(new_num_rows):
        for k in range(look_back):
            if k == 0:
                new_df_row = np.array(orig_df_reindexed[comp_var][j])
            new_df_row = np.hstack((new_df_row, orig_df_reindexed.iloc[i+j+k])) # slide rows horizontally
            if j == 0:
                new_df_colnames = new_df_colnames + [name+str(-i-k) for name in list(orig_df_reindexed)] # add to columns names
        if j == 0:
            new_df_array = new_df_row
        else:
            new_df_array = np.vstack((new_df_array, new_df_row)) # stack new rows
    if new_df_array.ndim == 1:
        new_df = pd.DataFrame(columns=new_df_colnames)
        new_df.loc[0] = new_df_array
    else:
        new_df = pd.DataFrame(data=new_df_array, columns=new_df_colnames) 
        # print(new_df)
    return new_df


def avg_percent_error(y_hat, y):
    return np.mean(np.abs((y_hat-y)/y))


def within_percent(a, b, percent):
    """Returns true if a is within some percent of b"""

    return percent >= 100*abs(a-b)/b


# Hopping Lasso Regression

# Optimize model at single offset
# def hyper_param_opt(train_df, model_type, offset, \
#     look_back_step=20, look_back_override=False, test_props=[.25,.2,.15,.1], \
#     iter=10, n_alphas=5, cv=3, eps=.001, comp_var='Price'):
    
#     max_samples = len(train_df)-offset-1 # If look_back were the minimum (1) we would have this many samples
#     min_samples = 24 # We want to have at least this many samples to do a train/test split
#     if look_back_override == 'single':
#         look_backs = [look_back_step]
#     elif look_back_override == 'max':
#         look_backs = [max_samples - min_samples]
#     else:
#         look_backs = list(np.arange(look_back_step, max_samples-min_samples, look_back_step))
#         # look_backs = list(np.arange(look_back_step, len(train_df)+offset, look_back_step))
#     num_tests = len(look_backs)*len(test_props)*iter
#     complete = 0
#     results = []
#     for look_back in look_backs:
#         slid_df = slide_df(train_df, offset, look_back, purpose='train',comp_var=comp_var)
#         # slid_df = trim_features(slid_df, look_back=0)
#         X = slid_df.drop((comp_var), axis=1).values
#         y = slid_df[comp_var].values
#         if len(y) >= 24:
#             for test_prop in test_props*iter:
#                 complete += 1
#                 print('Training offset: '+str(offset)+' with look_back: '+str(look_back)+', test_prop: '+str(np.round(test_prop,2))+', '+str(np.round(100*complete/num_tests, 2))+'% complete', end='      \r')
#                 model = model_type(n_alphas=n_alphas, cv=cv, eps=eps, normalize=True)
#                 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_prop)
#                 model.fit(X_train, y_train)
#                 train_error = avg_percent_error(model.predict(X_train),y_train)
#                 test_error = avg_percent_error(model.predict(X_test),y_test)
#                 results.append({'offset':offset,
#                                 'n_alphas':n_alphas,
#                                 'cv':cv,
#                                 'eps':eps,
#                                 'look_back':look_back,
#                                 'len_data':len(y),
#                                 'train_prop':np.round(1-test_prop,2),
#                                 'train_num':len(y_train),
#                                 'test_num':len(y_test),
#                                 'alpha':model.alpha_,
#                                 'train_error':train_error,
#                                 'test_error':test_error})
#         else:
#             num_tests -= 1
#     hp_df = pd.DataFrame(data=results)
#     return hp_df

def hyper_param_opt(train_df, offset,
    look_back_step=20, look_back_override=False, test_props=[.2,.15,.1],
    iter=10, model_type=Lasso, n_alphas=10, comp_var='Price'):
    
    # If look_back were the minimum (1) we would have max_samples
    # We want to have at least min_samples samples to do a train/test split
    max_samples, min_samples, min_test_points = len(train_df)-offset-1, 24, 12 # If look_back were the minimum (1) we would have this many samples
    if look_back_override == 'single':
        look_backs = [look_back_step]
    elif look_back_override == 'max':
        look_backs = [max_samples - min_samples]
    else:
        look_backs = list(np.flip(np.arange(look_back_step, max_samples-min_samples, look_back_step), axis=0))
        # look_backs = list(np.arange(look_back_step, len(train_df)+offset, look_back_step))
    num_tests = len(look_backs)*len(test_props)*n_alphas*iter
    complete, results = 0, []
    alphas = list(np.linspace(.0001,.0015, num=n_alphas))
    for look_back in look_backs:
        slid_df = slide_df(train_df, offset, look_back, purpose='train',comp_var=comp_var)
        # slid_df = trim_features(slid_df, look_back=0)
        X = slid_df.drop((comp_var), axis=1).values
        y = slid_df[comp_var].values
        feat_names = list(slid_df.drop((comp_var), axis=1))
        # print(feat_names)
        for test_prop in test_props:
            if len(y)*test_prop >= min_test_points:
                for alpha in alphas*iter:
                    model = model_type(alpha=alpha, normalize=True)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_prop)
                    model.fit(X_train, y_train)
                    train_error = avg_percent_error(model.predict(X_train),y_train)
                    test_error = avg_percent_error(model.predict(X_test),y_test)
                    # print(list(model.coef_>0))
                    kept_feats = list(itertools.compress(feat_names, list(model.coef_>0)))
                    # print(kept_feats)
                    results.append({'offset':offset,
                                    'look_back':look_back,
                                    'len_data':len(y),
                                    'train_prop':np.round(1-test_prop,2),
                                    'train_num':len(y_train),
                                    'test_num':len(y_test),
                                    'neg_alpha':np.round(-alpha, 6),
                                    'num_feats':sum(model.coef_>0),
                                    'kept_feats':'+'.join(kept_feats),
                                    'train_error':train_error,
                                    'test_error':test_error,
                                    'n_iter':model.n_iter_})
                    # coefs
                    complete += 1
                    print('Offset: '+str(offset)+' look_back: '+str(look_back)+' test_prop: '+str(np.round(test_prop,2))+' alpha: '+str(np.round(alpha,5))+'... '+str(np.round(100*complete/num_tests, 2))+'% complete', end='            \r')
            else:
                num_tests -= 1
    hp_df = pd.DataFrame(data=results)
    return hp_df


# Implement hopping lasso for many offsets
# def train_hopping_lasso(orig_df, comp_var='Price', pred_len=60, iter=50, \
#     look_back_step=12, look_back_override=False, test_props=[.2,.15,.1], \
#     n_alphas=5, cv=3, eps=.001):

#     df = orig_df.copy()

#     offsets = list(np.arange(1,pred_len+1))
#     offsets_plot = []
#     training_dfs = []
#     best_errors = []
#     best_hps = []

#     fig, ax = plt.subplots(1,1)

#     for offset in offsets:
#         hp_df = hyper_param_opt(df, LassoCV, offset, \
#                                 look_back_step=look_back_step, \
#                                 look_back_override=look_back_override, \
#                                 test_props=test_props, \
#                                 iter=iter, \
#                                 n_alphas=n_alphas, \
#                                 cv=cv, \
#                                 eps=eps, \
#                                 comp_var=comp_var) \
#                                 .groupby(['offset','n_alphas','cv','eps','alpha','look_back','train_prop']).mean()
#         offsets_plot.append(offset)
#         training_dfs.append(hp_df)
#         min_error = hp_df.test_error.min()
#         close_errors = hp_df[within_percent(hp_df.test_error, min_error, 10)]
#         best_errors.append(close_errors.test_error[0])
#         best_hps.append(close_errors.index[0])

#     print(best_hps)
#     ax.plot(offsets_plot, best_errors)
#     plt.show()

#     return (best_hps, training_dfs)


def train_hopping_lasso(orig_df, comp_var='Price', pred_len=60, iter=50,
    look_back_step=12, look_back_override=False, test_props=[.2,.15,.1],
    n_alphas=10, model_type=Lasso):

    df = orig_df.copy()

    offsets = list(np.arange(1,pred_len+1))
    offsets_plot = []
    training_dfs = []
    sample_dfs = []
    best_errors = []
    best_hps = []
    ceof_sets = []

    fig, ax = plt.subplots(1,1)

    for offset in offsets:
        offsets_plot.append(offset)
        full_hp_df = hyper_param_opt(df, offset,
                                look_back_step=look_back_step,
                                look_back_override=look_back_override,
                                test_props=test_props,
                                n_alphas=n_alphas,
                                iter=iter,
                                model_type=model_type,
                                comp_var=comp_var)
        mean_hp_df = full_hp_df.groupby(['offset','look_back','train_prop','neg_alpha']).mean()
        training_dfs.append(mean_hp_df)
        sample_df = full_hp_df.groupby(['offset','look_back','train_prop','neg_alpha']).last()
        sample_dfs.append(sample_df)
        min_error = mean_hp_df.test_error.min()
        close_errors = mean_hp_df[within_percent(mean_hp_df.test_error, min_error, 1)]
        best_errors.append(close_errors.test_error[0])
        best_hps.append(close_errors.index[0])

    print(best_hps)
    ax.plot(offsets_plot, best_errors)
    plt.show()

    return best_hps, training_dfs, sample_dfs


def var_name(x):
    try:
        for s, v in list(locals().iteritems()):
            if v is x:
                return s
    except Exception as e:
        print(e)


# Make predictions using optimized hyperparameters
# def hopping_lasso_predict(orig_df, best_hps, model_type=LassoCV, iter=100, comp_var='Price'):

    # df = orig_df.copy()

    # num_models = iter*len(best_hps)
    # i = 0

    # predictions = []
    # for hp_set in best_hps:
    #     (offset, n_alphas, cv, eps, alpha, look_back, train_size) =  hp_set
    #     hps = 
    #     offset = int(offset)
    #     look_back = int(look_back)
    #     models = [model_type(n_alphas=n_alphas, eps=eps, cv=cv, normalize=True)]*iter
    #     slid_df_train = slide_df(df, offset, look_back, purpose='train', comp_var=comp_var)
    #     # slid_df_train = trim_features(slid_df_train, look_back=0)
    #     X_train_complete = slid_df_train.drop((comp_var), axis=1).values
    #     y_train_complete = slid_df_train[comp_var].values
    #     slid_df_test = slide_df(df, 0, look_back, purpose='test', comp_var=comp_var)
    #     X_test = slid_df_test.drop((comp_var), axis=1).values
    #     for model in models:
    #         X_train_subset, not_used, y_train_subset, not_used2 = \
    #             train_test_split(X_train_complete, y_train_complete, test_size=1-train_size)
    #         model.fit(X_train_subset, y_train_subset)
    #         predictions.append({'offset':offset, 'prediction':np.round(model.predict(X_test)[0],3)})
    #         i += 1
    #         print(str(i)+' of '+str(num_models)+' models finished, predictions '+str(np.round(100*i/num_models, 2))+'% complete', end='        \r')

    # prediction_df = pd.DataFrame(data=predictions)
    # predictions = prediction_df.groupby('offset').mean().rename(columns={'prediction':'avg_prediction'})
    # stds = prediction_df.groupby('offset').std().rename(columns={'prediction':'std'})
    # predictions_df = pd.concat([predictions, stds], axis=1)
    # predictions_df['+std'] = predictions_df.avg_prediction+predictions_df['std']
    # predictions_df['-std'] = predictions_df.avg_prediction-predictions_df['std']

    # return predictions_df


def hopping_lasso_predict(orig_df, best_hps, iter=100, comp_var='Price'):

    df = orig_df.copy()

    num_models = iter*len(best_hps)
    i = 0

    predictions = []
    for hp_set in best_hps:
        (offset, look_back, train_size, neg_alpha) =  hp_set
        offset = int(offset)
        look_back = int(look_back)
        models = [Lasso(alpha=-neg_alpha, normalize=True)]*iter
        slid_df_train = slide_df(df, offset, look_back, purpose='train', comp_var=comp_var)
        # slid_df_train = trim_features(slid_df_train, look_back=0)
        X_train_complete = slid_df_train.drop((comp_var), axis=1).values
        y_train_complete = slid_df_train[comp_var].values
        slid_df_test = slide_df(df, 0, look_back, purpose='test', comp_var=comp_var)
        X_test = slid_df_test.drop((comp_var), axis=1).values
        for model in models:
            X_train_subset, not_used, y_train_subset, not_used2 = \
                train_test_split(X_train_complete, y_train_complete, test_size=1-train_size)
            model.fit(X_train_subset, y_train_subset)
            predictions.append({'offset':offset, 'prediction':np.round(model.predict(X_test)[0],3)})
            i += 1
            print(str(i)+' of '+str(num_models)+' models finished, predictions '+str(np.round(100*i/num_models, 2))+'% complete', end='        \r')

    prediction_df = pd.DataFrame(data=predictions)
    predictions = prediction_df.groupby('offset').mean().rename(columns={'prediction':'avg_prediction'})
    stds = prediction_df.groupby('offset').std().rename(columns={'prediction':'std'})
    predictions_df = pd.concat([predictions, stds], axis=1)
    predictions_df['+std'] = predictions_df.avg_prediction+predictions_df['std']
    predictions_df['-std'] = predictions_df.avg_prediction-predictions_df['std']

    return predictions_df




















