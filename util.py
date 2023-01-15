# import libraries
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

sns.set(style="whitegrid")
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_predict, train_test_split, KFold, StratifiedKFold, RepeatedKFold
from sys import stdout
import os
import logging
import configparser


# list files from current dir and filter with .csv
# all_files = os.listdir()
# csv_files = list(filter(lambda f: f.endswith('.csv'), all_files))


def get_config_values(object_type: str, list_type=True, key=None) -> list:
    """ Reads the specified config.ini object and returns a list of values """

    config_obj = configparser.ConfigParser()
    config_obj.read("configfile.ini")
    if list_type:
        values = []
        objects = config_obj[object_type]
        for key in objects:
            values.append(objects[key])
    else:
        values = config_obj[object_type][key]

    return values


# def train_validation_split(X_data, y_data, train_size):
#     """ Takes in X_train, y_train and splits it further into training + validation
#
#     @params -
#     X_data, y_data, train_size, validation_size
#     # returns - X_train, X_validation, y_train, y_validation
#     """
#     row_start_index_train, row_stop_index_train = 0, round(train_size * X_data.shape[0])
#     row_start_index_validation, row_stop_index_validation = row_stop_index_train + 1, X_data.shape[0]
#
#     X_train = X_data.iloc[row_start_index_train:row_stop_index_train,:]
#     y_train = y_data.iloc[row_start_index_train:row_stop_index_train]
#     X_validation = X_data.iloc[row_start_index_validation:row_stop_index_validation,:]
#     y_validation = y_data.iloc[row_start_index_validation:row_stop_index_validation]
#
#     return X_train, X_validation, y_train, y_validation

def data_eda(df: pd.DataFrame):
    print('Shape of the dataset : ' + str(df.shape))
    print('Dataset overview : \n')
    print(df.head())
    print('Dataset description : \n')
    print(df.describe())
    print('Null values in data : ' + str(df.isnull().sum()))


def get_split_data(files: list, split_params: dict, test_size: float, showDataEDA: bool) -> tuple:
    """ Data split into required training / testing datasets

    @Params:
    split_params   contains details like {sep, index_column, etc.}
    files          --> split operation and distribution of result dataset will depend on index of supplied files array
    test_size      the ratio for train-test split

    # returns - X_train, X_test, y_train, y_test
    """

    # Check if multiple files were supplied
    if type(files) == list:  # multiple files
        # read all files and create dataframe
        df_list = []
        for i, file in enumerate(files):
            data = pd.read_csv(file, sep=split_params['sep'])
            data.set_index(split_params['index_column'])
            df_list.append(data)
        # concatenate them together
        big_df = pd.concat(df_list, ignore_index=True)
        X = big_df.drop(axis=1,
                        columns=split_params['drop_columns'])  # drop the last column, so we have exactly 256 features
        y = big_df[split_params['y_data_column']]

        if showDataEDA:
            # Print data EDA
            data_eda(big_df)

        # when shuffle = false means extrapolation in below code
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True)
        return X_train, X_test, y_train, y_test

    else:  # single file
        # read the file and create dataframe
        data = pd.read_csv(files, sep=split_params['sep'])
        data.set_index(split_params['index_column'])
        X = data.drop(axis=1,
                      columns=split_params['drop_columns'])  # drop the last column, so we have exactly 256 features
        y = data[split_params['y_data_column']]

        if showDataEDA:
            # Print data EDA
            data_eda(data)

        '''  ** MANUAL SPLITTING CODE **   [when not using the inbuilt stratification]
            # split data based on timestamp to capture a chunk in the beginning as train set
            row_start_index_train, row_stop_index_train = 0, (training_split / 100) * X.shape[0]
            row_start_index_validation, row_stop_index_validation = 0, (validation_split / 100) * X.shape[0]
            row_start_index_test, row_stop_index_test = 0, (test_split / 100) * X.shape[0]

            X_train = X.iloc[row_start_index_train:row_stop_index_train,:]
            y_train = y.iloc[row_start_index_train:row_stop_index_train]
            X_validation = X.iloc[row_start_index_validation:row_stop_index_validation,:]
            y_validation = y.iloc[row_start_index_validation:row_stop_index_validation]
            X_test = X.iloc[row_start_index_test:row_stop_index_test,:]
            y_test = y.iloc[row_start_index_test:row_stop_index_test]
            '''
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True)
        return X_train, X_test, y_train, y_test


def pipeline(pipeline_params: dict):
    """ The main method for all ML Pipelines -
    executes all steps (pre-processing + training, validation, and then testing)
    in various combinations as per @params

    @Params:
    pipeline_params = {split_type, split_params, preprocessor, model, evaluator, sep, drop_column ...}

    returns: None (stores all output to the results logs file for each run)
    """

    logger = pipeline_params['logger']
    showPlots = pipeline_params['plotting']
    showModelEvaluationPlots = pipeline_params['showModelEvaluationPlots']
    model = pipeline_params['model']
    evaluator = pipeline_params['evaluator']
    wl = pipeline_params['wl']
    folds = pipeline_params['folds']
    pipeline_run_type = pipeline_params['pipeline_run_type']

    # Get the data
    data = (X_train, X_test, y_train, y_test) = pipeline_params['data']

    if showPlots:
        # plot raw training spectrum
        plot(x_axis_data=wl, y_axis_data=X_train, title='Spectrum plot of training data', xlabel='wavelength',
             ylabel='absorbance')

    # Preprocess + train -> then validate / hyperparameter tuning -> then test / evaluate score on test set

    pre_processor = pipeline_params['preprocessor']
    pre_processor_params = pipeline_params['preprocessor-params']

    # Check model selected
    if model == 'pls':

        # [ STEP 1 ] Check type of preprocessor and optimise PLS accordingly
        logger.info("[PRE PROCESSING]")

        pls_max_n_comp = pipeline_params['pls_max_n_comp']

        if pre_processor == 'savgol0' or pre_processor == 'savgol1' or pre_processor == 'savgol2':
            window_range = pre_processor_params['savgol']['window_size_range']
            polyorder_range = pre_processor_params['savgol']['polyorder_range']
            derivative = pre_processor_params['savgol']['derivative']

            X_train_preprocessed, X_test_preprocessed, least_mse_score, best_window_size, best_polyorder = \
                savgol(data, window_size_range=window_range,
                       polyorder_range=polyorder_range,
                       deriv=derivative, showPlot=showPlots)

            # SAVE LEAST MSE_SCORE VALUE, BEST WINDOW_SIZE AND BEST POLYNOMIAL INTO RESULTS
            logger.info("Least MSE from SAVGOL loop run {}".format(least_mse_score))
            logger.info("Best window size {}".format(best_window_size))
            logger.info("Best polyorder {}".format(best_polyorder))
            # print("Least MSE from SAVGOL loop run : "+str(least_mse_score))
            if showPlots:
                plot(x_axis_data=wl, y_axis_data=X_train_preprocessed,
                     title='Post SavGol plot (Deriv : ' + str(derivative) + ' )', xlabel='wavelength',
                     ylabel='absorbance')
        elif pre_processor == 'snv':
            X_train_preprocessed, X_test_preprocessed = snv(data)
            if showPlots:
                plot(x_axis_data=wl, y_axis_data=X_train_preprocessed, title='Post SNV Plot', xlabel='wavelength',
                     ylabel='absorbance')
        elif pre_processor == 'msc':
            X_train_preprocessed, X_test_preprocessed = msc(data)
            if showPlots:
                plot(x_axis_data=wl, y_axis_data=X_train_preprocessed, title='Post MSC plot', xlabel='wavelength',
                     ylabel='absorbance')
        elif pre_processor == 'savgol0+snv' or pre_processor == 'savgol1+snv' or pre_processor == 'savgol2+snv':
            window_range = pre_processor_params['savgol']['window_size_range']
            polyorder_range = pre_processor_params['savgol']['polyorder_range']
            derivative = pre_processor_params['savgol']['derivative']

            X_train_preprocessed_1, X_test_preprocessed_1, least_mse_score, best_window_size, best_polyorder = \
                savgol(data, window_size_range=window_range,
                       polyorder_range=polyorder_range,
                       deriv=derivative, showPlot=showPlots)
            data_1 = X_train_preprocessed_1, X_test_preprocessed_1, y_train, y_test
            X_train_preprocessed, X_test_preprocessed = snv(data_1, afterSmoothing=True)
            if showPlots:
                plot(x_axis_data=wl, y_axis_data=X_train_preprocessed, title='Post SNV Plot', xlabel='wavelength',
                     ylabel='absorbance')
        elif pre_processor == 'savgol0+msc' or pre_processor == 'savgol1+msc' or pre_processor == 'savgol2+msc':
            window_range = pre_processor_params['savgol']['window_size_range']
            polyorder_range = pre_processor_params['savgol']['polyorder_range']
            derivative = pre_processor_params['savgol']['derivative']

            X_train_preprocessed_1, X_test_preprocessed_1, least_mse_score, best_window_size, best_polyorder = \
                savgol(data, window_size_range=window_range,
                       polyorder_range=polyorder_range,
                       deriv=derivative, showPlot=showPlots)
            data_1 = X_train_preprocessed_1, X_test_preprocessed_1, y_train, y_test
            X_train_preprocessed, X_test_preprocessed = msc(data_1, afterSmoothing=True)
            if showPlots:
                plot(x_axis_data=wl, y_axis_data=X_train_preprocessed, title='Post MSC plot', xlabel='wavelength',
                     ylabel='absorbance')

        if pipeline_run_type != 'preprocessor':

            # [ STEP 2 ] Validation and hyperparameter (n_comp of PLS) optimisation

            logger.info("[VALIDATION AND HYPERPARAMETER (N_COMP) TUNING]")

            # Check selected validation type
            # SYNTAX - value_when_true if condition else value_when_false
            validation_type = 'kfold-cv' if pipeline_params['validation-type'] == 'kfold-cv' else 'stratified-kfold-cv'
            validation_params = {
                'validation_type': validation_type,
                'folds': folds,
                'pls_max_n_comp': pls_max_n_comp
            }

            best_n_comp_val = optimize_and_validate_pls(X_train_preprocessed, y_train, validation_params,
                                                        showPlot=showPlots)
            # SAVE OPTIMISED HYPERPARAMETER VALUE
            logger.info("Optimized n_comp for PLS {}".format(best_n_comp_val))

            # [ STEP 3 ] Model evaluation
            logger.info("[MODEL EVALUATION]")
            error = evaluate_pls(X_train_preprocessed, X_test_preprocessed, y_train, y_test, best_n_comp_val,
                                 showPlot=True, showModelEvaluationPlots=showModelEvaluationPlots)
            logger.info("Model loss : {}".format(error))

    elif model == 'lasso':
        alpha_range = pipeline_params['alpha_range']

        # define model
        model = Lasso(alpha=1.0)

        # [ Step 1 ] Fit model on raw spectrum and check initial results
        model.fit(X_train, y_train)

        # make a prediction
        y_pred = model.predict(X_test)

        # Calculate mean squared error for y_pred and y_test
        mse_c = mean_squared_error(y_test, y_pred)
        print('[Model Evaluation] MSE value between y_test and y_pred : %5.3f' % mse_c)

        # [ Step 2 ] - Apply preprocessing (SNV / MSC) and redo Step 1
        if pre_processor == 'snv':
            X_train_preprocessed, X_test_preprocessed = snv(data)
            if showPlots:
                plot(x_axis_data=wl, y_axis_data=X_train_preprocessed, title='Post SNV Plot', xlabel='wavelength',
                     ylabel='absorbance')
        elif pre_processor == 'msc':
            X_train_preprocessed, X_test_preprocessed = msc(data)
            if showPlots:
                plot(x_axis_data=wl, y_axis_data=X_train_preprocessed, title='Post MSC plot', xlabel='wavelength',
                     ylabel='absorbance')

        # Fit model again on preprocessed data and check results
        model.fit(X_train_preprocessed, y_train)
        # plt.figure()
        # weights = model.coef_
        # plt.plot(weights)

        # make a prediction
        y_pred = model.predict(X_test_preprocessed)

        # plt.figure()
        # plt.plot(y_test)
        # plt.plot(y_pred)
        # plt.show()

        # Calculate mean squared error for y_pred and y_test
        mse_c = mean_squared_error(y_test, y_pred)
        print('[Model Evaluation] MSE value between y_test and y_pred after pre-processing : %5.3f' % mse_c)

        # [ Step 3 ] - Model tuning (adjusting the value of alpha) using CV
        logger.info("[VALIDATION AND HYPERPARAMETER (Alpha value) TUNING]")

        # Check selected validation type
        # SYNTAX - value_when_true if condition else value_when_false
        # validation_type = 'kfold-cv' if pipeline_params['validation-type'] == 'kfold-cv' else 'stratified-kfold-cv'
        validation_params = {
            'validation_type': pipeline_params['validation-type'],
            'folds': folds,
            'alpha_range': alpha_range
        }

        # best_alpha_val = optimize_and_validate_lasso(X_train, y_train, validation_params, showPlot=showPlots)
        best_alpha_val = optimize_and_validate_lasso(X_train_preprocessed, X_test_preprocessed, y_train, y_test,
                                                     validation_params, showPlot=showPlots)
        # SAVE OPTIMIZED HYPERPARAMETER VALUE
        logger.info("Optimized alpha value for Lasso {}".format(best_alpha_val))

        # [ STEP 4 ] Model evaluation
        logger.info("[MODEL EVALUATION]")
        error = evaluate_lasso(X_train_preprocessed, X_test_preprocessed, y_train, y_test, best_alpha_val,
                               showPlot=True, showModelEvaluationPlots=showModelEvaluationPlots)
        # error = evaluate_lasso(X_train, X_test, y_train, y_test, best_alpha_val, showPlot=True)
        logger.info("Model loss : {}".format(error))


def savgol(data, window_size_range, polyorder_range, deriv=0, showPlot=False):
    """ Step 1: Implement Sav-Gol in a loop to produce a heatmap showing optimum values for w (window size) and p (no. of polynomials)
             based on lowest RMSE value
        Step 2: Run Sav-Gol with the optimum hyper-parameters, plot the new X_data and return it

    @Params:
    data = contains X_train, X_test, y_train, y_test
    window_size_range = range of window_size values to try out savgol
    polyorder_range = range of polyorder values to try out savgol
    deriv = (default is 0) derivative order of savgol operation

    # returns - X_train_smooth, X_test_smooth (smoothened spectrum data)
    """
    X_train, X_test, y_train, y_test = data

    mse_scores = np.zeros(shape=(20, 8))
    for index_w, w in enumerate(window_size_range):
        for index_p, p in enumerate(polyorder_range):
            # polyorder must be less than window_length to calculate value
            if p < w:
                X_smooth = savgol_filter(X_train, w, polyorder=p, deriv=deriv)
                pls1model = PLSRegression(n_components=2)
                pls1model.fit(X_smooth, y_train)
                y_pred = pls1model.predict(X_smooth)
                # RMSE regression loss.Returns loss - (best value is 0.0)
                mse_scores[index_w][index_p] = mean_squared_error(y_train, y_pred)

    # Plot
    if showPlot:
        range_min = round(np.amin(mse_scores[mse_scores != 0.0]) - 0.3, 1)
        range_max = round(np.amax(mse_scores[mse_scores != 0.0]) + 0.3, 1)
        heatmap_plot(mse_scores, range_min, range_max, polyorder_range, window_size_range,
                     'MSE scores after SAVGOL  (Deriv : ' + str(deriv) + ' )', 'Polyorder', 'Window size')

    ''' Perform savgol with optimum window_size and polyorder '''
    # min_mse_val_coord = np.where(mse_scores == np.amin(mse_scores[mse_scores != 0.0]))      # --> returns (array([0]), array([0]))
    least_mse_score = np.amin(mse_scores[mse_scores != 0.0])
    min_mse_val_coord = np.argwhere(mse_scores == np.amin(mse_scores[mse_scores != 0.0]))  # --> returns array([0,0])
    # np.unravel_index(np.argmin(mse_scores[mse_scores != 0], axis=None), mse_scores.shape)
    best_window_size, best_polyorder = window_size_range[min_mse_val_coord[0][0]], polyorder_range[
        min_mse_val_coord[0][1]]

    X_train_smooth = savgol_filter(X_train, best_window_size, polyorder=best_polyorder, deriv=deriv)
    X_test_smooth = savgol_filter(X_test, best_window_size, polyorder=best_polyorder, deriv=deriv)

    return X_train_smooth, X_test_smooth, least_mse_score, best_window_size, best_polyorder


def snv(data, afterSmoothing=False):
    """ Method takes in an input_X_data (spectrum data or smoothed data for example) and outputs
    SNV (standard normal variate) processed data

    @Params:
    data - (e.g. spectrum data or smoothed data)

    # returns - X_train_preprocessed, X_test_preprocessed
    """
    X_train, X_test, y_train, y_test = data
    if not afterSmoothing:
        X_train = X_train.values
        X_test = X_test.values

    # Define a new array and populate it with the corrected data
    X_train_preprocessed = np.zeros_like(X_train)
    X_test_preprocessed = np.zeros_like(X_test)
    for i in range(X_train.shape[0]):
        # Apply correction
        X_train_preprocessed[i, :] = (X_train[i, :] - np.mean(X_train[i, :])) / np.std(X_train[i, :])
    for i in range(X_test.shape[0]):
        # Apply correction
        X_test_preprocessed[i, :] = (X_test[i, :] - np.mean(X_test[i, :])) / np.std(X_test[i, :])

    return X_train_preprocessed, X_test_preprocessed


def msc(data, train_reference=None, test_reference=None, afterSmoothing=False):
    """ Method takes input data (spectrum data or smoothed data for example) and outputs
    MSC (Multiplicative Scatter Correction) processed data

    @Params:
    data - (e.g. spectrum data or smoothed data)

    # returns - X_train_preprocessed, X_test_preprocessed
    """
    X_train, X_test, y_train, y_test = data
    if not afterSmoothing:
        X_train = X_train.values
        X_test = X_test.values

    # mean centre correction
    for i in range(X_train.shape[0]):
        X_train[i, :] -= X_train[i, :].mean()
    for i in range(X_test.shape[0]):
        X_test[i, :] -= X_test[i, :].mean()

    # Get the reference spectrum. If not given, estimate it from the mean
    if train_reference is None:
        # Calculate mean
        train_ref = np.mean(X_train, axis=0)
    else:
        train_ref = train_reference

    if test_reference is None:
        # Calculate mean
        test_ref = np.mean(X_test, axis=0)
    else:
        test_ref = test_reference

    # Define a new array and populate it with the corrected data
    X_train_preprocessed = np.zeros_like(X_train)
    for i in range(X_train.shape[0]):
        # Run regression
        fit = np.polyfit(train_ref, X_train[i, :], 1, full=True)
        # Apply correction
        X_train_preprocessed[i, :] = (X_train[i, :] - fit[0][1]) / fit[0][0]

    X_test_preprocessed = np.zeros_like(X_test)
    for i in range(X_test.shape[0]):
        # Run regression
        fit = np.polyfit(test_ref, X_test[i, :], 1, full=True)
        # Apply correction
        X_test_preprocessed[i, :] = (X_test[i, :] - fit[0][1]) / fit[0][0]

    return X_train_preprocessed, X_test_preprocessed


def optimize_and_validate_lasso(X_train, X_test, y_train, y_test, validation_params, showPlot=False):
    # define model evaluation method

    if validation_params['validation_type'] == 'kfold-cv':
        cv = KFold(n_splits=validation_params['folds'], random_state=None, shuffle=False)
    elif validation_params['validation_type'] == 'stratified-kfold-cv':
        cv = StratifiedKFold(n_splits=validation_params['folds'], random_state=None, shuffle=False)
    elif validation_params['validation_type'] == 'repeated-kfold-cv':
        cv = RepeatedKFold(n_repeats=2, n_splits=validation_params['folds'],
                           random_state=2652124)  # any int, helps reproduce same split over multiple runs
    # define model
    # alphas=np.arange(0, 1, 0.01)
    model = LassoCV(cv=cv, random_state=0, max_iter=10000)
    # fit model
    model.fit(X_train, y_train)

    # make a prediction
    y_pred = model.predict(X_test)

    # plt.figure()
    # weights = model.coef_
    # plt.plot(weights)
    #
    # plt.figure()
    # # plt.plot(y_test,'.')
    # # plt.plot(y_pred,'.')
    # print(y_pred.shape)
    # print(y_test.shape)
    # plt.show()

    plt.figure(figsize=(8, 5))
    with plt.style.context('ggplot'):
        plt.plot(y_test, '.')
        plt.plot(y_pred, '.')
        plt.show()

    # summarize chosen configuration
    return model.alpha_


def optimize_and_validate_pls(X, y, validation_params, showPlot=False):
    """ Run PLS while tuning n_comp (hyperparameter) such that MSE is least AS PER THE SELECTED VALIDATION TECHNIQUE

     @params
     X -> X_train_preprocessed,
     y -> y_train,
     validation_params,
     showPlot

     # returns best n_components value for PLS

     """
    validation_type = validation_params['validation_type']
    pls_max_n_comp = validation_params['pls_max_n_comp']
    folds = validation_params['folds']
    mse = []
    component = np.arange(1, pls_max_n_comp)

    for i in component:
        pls = PLSRegression(n_components=i)

        ''' cross-val-predict internally splits data into training and validation, as per the provided "folds" value 
        (if int value supplied then StratifiedKFolds method, else KFolds and default folds value=5) 
        
        cross-val-predict returns y_pred (by averaging from all iterations)
        '''

        # Check if KFolds or StratifiedKFolds
        cv = folds if validation_type == 'stratified-kfold-cv' else None

        # Cross-validation
        y_cv = cross_val_predict(pls, X, y, cv=cv)  # int means that no. of fold in StratifiedKFold form

        mse.append(mean_squared_error(y, y_cv))

        # Show completion %
        comp = 100 * (i + 1) / 40
        # Trick to update status on the same line
        stdout.write("\r%d%% Validation completed " % comp)
        stdout.flush()
    stdout.write("\n")

    # Calculate and print the position of minimum in MSE
    msemin = np.argmin(mse)
    stdout.write("\n")

    if showPlot:
        with plt.style.context(('ggplot')):
            plt.plot(component, np.array(mse), '-v', color='blue', mfc='blue')
            plt.plot(component[msemin], np.array(mse)[msemin], 'P', ms=10, mfc='red')
            plt.xlabel('Number of PLS components')
            plt.ylabel('MSE')
            plt.title('PLS')
            plt.xlim(left=-1)

        plt.show()

    return msemin + 1


def evaluate_pls(X_train, X_test, y_train, y_test, min_pls_n_comp, showPlot=False, showModelEvaluationPlots=True):
    """ Run PLS with min_pls_n_comp number of components and
    calculate MSE between y_pred and y_test

     @params
     X_test
     y_test
     min_pls_n_comp

     returns MSE between y_pred and y_test

     """

    # Define PLS object with optimal number of components
    pls_opt = PLSRegression(n_components=min_pls_n_comp)

    # Fit to training set with the best hyperparameter value
    pls_opt.fit(X_train, y_train)

    # Now predict on test set
    y_pred = pls_opt.predict(X_test)

    # Calculate mean squared error for y_pred and y_test
    mse_c = mean_squared_error(y_test, y_pred)

    # SAVE MSE VALUE CALCULATED BETWEEN Y_TEST AND Y_PRED
    print('[Model Evaluation] MSE value between y_test and y_pred : %5.3f' % mse_c)

    # Plot regression and figures of merit
    rangey = max(y_test) - min(y_test)
    rangex = max(y_pred) - min(y_pred)

    if showModelEvaluationPlots:
        # Fit a line to the test vs pred
        z = np.polyfit(y_test, y_pred, 1)
        with plt.style.context(('ggplot')):
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.scatter(y_pred, y_test, c='red', edgecolors='k')
            # Plot the best fit line
            ax.plot(np.polyval(z, y_test), y_test, c='blue', linewidth=1)
            # Plot the ideal 1:1 line
            ax.plot(y_test, y_test, color='green', linewidth=1)
            # plt.title('$R^{2}$ (CV): '+str(score_cv))
            plt.xlabel('Predicted $^{\circ}$Brix')
            plt.ylabel('Measured $^{\circ}$Brix')

            plt.show()

    if showModelEvaluationPlots:
        pass
        '''# Plot y_data vs y_pred
        with plt.style.context(('ggplot')):
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(y_test, 'r')
            ax.plot(y_pred, 'g')
            plt.xlabel('Time')
            plt.ylabel('TS_sim')

            plt.show()'''

    return mse_c


def evaluate_lasso(X_train, X_test, y_train, y_test, best_alpha_val, showPlot=False, showModelEvaluationPlots=True):
    # Set best alpha
    lasso_best = Lasso(alpha=best_alpha_val)
    lasso_best.fit(X_train, y_train)

    # EVALUATION
    loss = mean_squared_error(y_test, lasso_best.predict(X_test))

    return loss


def result(path, file):
    """[Create a result file to record the experiment's results]

    @params:
        path {string} -- path to the directory
        file {string} -- file name

    Returns:
        [obj] -- [logger that record logs]
    """

    # check if the file exist
    result_file = os.path.join(path, file)

    if not os.path.isfile(result_file):
        open(result_file, "w+").close()

    console_logging_format = "%(levelname)s %(message)s"
    file_logging_format = "%(levelname)s: %(asctime)s: %(message)s"

    # configure logger
    logging.basicConfig(level=logging.INFO, format=console_logging_format)
    logger = logging.getLogger()

    # create a file handler for output file
    handler = logging.FileHandler(result_file)

    # set the logging level for log file
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter(file_logging_format)
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    logger.propagate = False

    return logger


def plot(x_axis_data, y_axis_data, title='', xlabel='', ylabel=''):
    """ Produces a simple plot based on values from x_axis_data and y_axis_data

    @Params:
    x_axis_data
    y_axis_data
    title
    x_label
    y_label
    """

    plt.figure(figsize=(8, 5))
    with plt.style.context('ggplot'):
        plt.plot(x_axis_data, y_axis_data.T)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()


def heatmap_plot(score_values, range_min, range_max, x_axis_data, y_axis_data, title='', xlabel='', ylabel=''):
    """ Produces a heatmap based on values from "score_values"

    @Params:
    score_values = data on which the heatmap is generated
    range_min = value for sns.heatmap vmin
    range_max = value for sns.heatmap vmax
    x_axis_data
    y_axis_data
    title
    x_label
    y_label
    """

    # Plot
    df = pd.DataFrame(score_values, columns=x_axis_data, index=y_axis_data)
    f, ax = plt.subplots(figsize=(10, 7))
    ax = sns.heatmap(df, vmin=range_min, vmax=range_max, cmap="Blues", linewidth=0.5, annot=True, fmt=".3f")
    ax.set(title=title)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.show()

# def plot_3D(x, y, z, xlabel, ylabel, zlabel, title):
#
#     fig = plt.figure(figsize=(9, 6))
#     ax = plt.axes(projection = '3d')
#
#     # 3d scatter plot
#     ax.plot3D(x, y, z)
#
#     #give labels
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     ax.set_zlabel(zlabel)
#     plt.title(title)
