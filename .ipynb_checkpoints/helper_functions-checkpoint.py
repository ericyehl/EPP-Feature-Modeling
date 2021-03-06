## Functions supporting the Project notebook

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random as rnd
import gspread
from oauth2client.service_account import ServiceAccountCredentials


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
    map, ax = plt.subplots(figsize=(num_cols/3,num_cols/3))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(np.round(corr,2), mask=mask, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=0, annot=include_annot, annot_kws={"size": 7}, cbar_kws={"shrink":.25})
    return map

 
# Pearson coeff
def pearson(sig1,sig2):
    """Calculate the Pearson Correlation Coefficient of two signals (two numpy vectors)"""

    return np.cov(sig1,sig2)[0,1]/(np.std(sig1)*np.std(sig2))


# Time heatmap
def time_heat_map(orig_df, comp_var='Price', months=60, include_annot=False):
    """Create a heatmap of Pearson correlation coefficients where all features are compared to one feature at different time delays

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
            time_corr[i,j] = np.round(pearson(comp_var_sig,feature_sig,2)

    # Set up the matplotlib figure
    map, ax = plt.subplots(figsize=(months/3,num_cols/3))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Create dataframe from matrix of correlations
    time_corr_df = pd.DataFrame(data=np.transpose(time_corr),columns = feature_names)
    time_corr_df = time_corr_df.reindex_axis(time_corr_df.abs().max().sort_values().index, axis=1)
    #time_corr_df = time_corr_df.reindex_axis(time_corr_df.mean().sort_values().index, axis=1)

    # Generate heat map
    sns.heatmap(time_corr_df.transpose(), cmap=cmap, vmax=1, center=0,
                square=False, linewidths=0, annot=include_annot, annot_kws={"size": 7}, cbar_kws={"shrink":.25})
    return map


# Construct new features with function transformations
def new_features_with_funcs(orig_df, funcs, func_names):
    """Create new feature columns out of old feature columns by applying transforming functions
     
    Keyword arguments:
    orig_df: pandas dataframe
    funcs: list of functions to apply to data
	func_names: list of names of functions to apply to data (for naming new columns)
    """

    df = orig_df.copy()

    for col_name in list(df):
        for i in range(len(funcs)):
            df[col_name+'_'+func_names[i]] = funcs[i](df[col_name])
    return df


# Construct new features with combination
def new_features_with_combs(orig_df, comp_var='Price'):
    """Create new feature columns out of old feature columns by combining features with multiplication.
    If a new feature is poorly correlated with price (incl any time shift up to 7 months), throw it out.
     
    Keyword arguments:
    orig_df: pandas dataframe
    comp_var: name of feature to try out correlations of new features
    """

    df = orig_df.copy()
    col_names = list(df)

    new_feats = 0
    thrown_out = 0


    for col_name1 in col_names:
        for col_name2 in col_names:
            new_feature = df[col_name1].multiply(df[col_name2])
            good_corr = False
            pearson_coeffs = []
            for i in range(7): # Look back 7 months for good correlation
                comp_var_sig = df[comp_var][i:len(df[comp_var])+1]
                new_feature_sig = new_feature[0:len(new_feature)-i]
                pearson_coeffs[i] = pearson(comp_var_sig,new_feature_sig)
            good_corr = np.max(np.abs(pearson_coeffs)) >= 0.6
            if good_corr:
                df[col_name1+'*'+col_name2] = df[col_name1].multiply(df[col_name2])
                new_feats += 1
            else:
                thrown_out += 1
    return df,new_feats,thrown_out

# Find names of highest absolute value correlations
