import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
from finite_ntk import utils
import matplotlib.lines as mlines
import matplotlib.colors as colors
from matplotlib import scale as mcscale
from matplotlib import transforms as mtransforms
from sklearn.metrics import mean_squared_error
import torch
import gpytorch
import argparse
import time
import math
import pickle
import sys
import finite_ntk
from sklearn.preprocessing import StandardScaler

class ExactGPModel(gpytorch.models.ExactGP):
    # exact RBF Gaussian process class
    def __init__(self, train_x, train_y, likelihood, model=None, use_linearstrategy=False, use_rbf_kernel=False):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        if use_rbf_kernel:
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.size(1))) #note the ard_num_dims argument
        else:
            self.covar_module = finite_ntk.lazy.NTK(
                model=model, use_linearstrategy=use_linearstrategy
            )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def load_data():
    # LOAD AND PREPROCESS THE REAL TRANSFER TASK DATA (ELECTRICITY CONSUMPTION)
    Y_partial = pd.read_csv("/work/GEFCOM2012_Data/Load/Load_history.csv")
    Y_additional = pd.read_csv("/work/GEFCOM2012_Data/Load/Load_solution.csv")
    X = pd.read_csv("/work/GEFCOM2012_Data/Load/temperature_history.csv")
    print(f"Shape of Y: {Y_partial.shape}")
    print(f"Shape of Y additional: {Y_additional.shape}")
    print(f"Shape of X: {X.shape}")
    print(f"Unique years: {X['year'].unique()}")
    print(f"Unique zones: {Y_partial['zone_id'].unique()}")
    print(f"Unique zones in additional Y: {Y_additional['zone_id'].unique()}")
    print(f"Unique temperature stations: {X['station_id'].unique()}")
    # 21th zone in the solution Y is the sum of other zones' load, so we drop such rows
    Y_additional = Y_additional[Y_additional['zone_id']!=21]

    # Original kaggle competition (where data is from) intentionally removed (i.e. set to NA) the temperature observations (Y) for some dates
    # The solutions for these dates are in Load_solution, so we need to fill the NAs in Load_history with the corresponding values from Load_solution
    Y = Y_partial.merge(Y_additional.drop(columns = ['weight']), how = 'left', on = ['zone_id', 'year', 'month', 'day'], suffixes = (None, "_additional"))
    hour_cols = [f'h{i}' for i in range(1, 25)]
    hour_cols_additional = [f'h{i}_additional' for i in range(1, 25)]
    for c, ca in zip(hour_cols, hour_cols_additional):
        Y[c].fillna(Y[ca], inplace=True)
    Y = Y.drop(columns = hour_cols_additional + ['id'])
    # Remove commas from numbers, and convert to numeric
    Y = Y.replace(',','', regex=True)
    c = Y.select_dtypes(object).columns
    Y[c] = Y[c].apply(pd.to_numeric)
    Y['datetime'] = pd.to_datetime(Y[['year', 'month', 'day']])
    # Some values are null after this date
    Y = Y[Y['datetime'] <= '2008-06-29'] 
    print(f"Shape of processed Y: {Y.shape}")
    print(f"Columns of processed Y with null rows: {Y.columns[Y.isnull().sum(axis = 0)!=0]}")
    return X, Y

# Newer version: allow feeding list of years
def get_data(X_df, Y_df, year_list, zone_id, n_points = None, seed = 1, hour = 8):
    '''
    return torch Tensors with n_points, filtered to year_list, zone_id, and hour (between 1 and 24)
    X_df: observations of temperature at 11 weather stations, at hour "hour""
    Y_df: observation of electricity consumption at zone_id, at hour "hour"
    '''
    hour_column = f"h{hour}"
    y_filt = Y_df[(Y_df['zone_id'] == zone_id)&(Y_df['year'].isin(year_list))][['year', 'month', 'day', hour_column]]
    X_filt = X_df[X_df['year'].isin(year_list)].pivot_table(index=['year', 'month', 'day'],columns='station_id',values=hour_column).reset_index()
    data = y_filt.merge(X_filt, how='left', on=['year','month', 'day']).drop(columns = ['year', 'month', 'day'])

    y = torch.from_numpy(data[hour_column].values).float()
    X = torch.from_numpy(data[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]].values).float()

    if n_points:
        assert(n_points <= len(X))
        np.random.seed(seed)
        idx = np.random.permutation(range(len(X)))[:n_points]
        return X[idx], y[idx]
    else: # return full filtered dataset if not specified
        return X, y

#### NN MODEL DEFINITION ####
def gen_model_specific(n_input_features, n_output_features, nhid = 40):
    model = torch.nn.Sequential(
        torch.nn.Linear(n_input_features, nhid),
        torch.nn.Tanh(),
        torch.nn.Linear(nhid, nhid),
        torch.nn.Tanh(),
        torch.nn.Linear(nhid, n_output_features)
    )
    print(f"Number of NN model params: {sum(p.numel() for p in model.parameters())}")
    return model

def get_results(train_x, train_y, transfer_x, transfer_y, test_x, test_y, n_input_features, n_output_features, nhid=40, method="untrained", 
                use_linearstrategy = False, iterations = 2500, num_batches_per_epoch = 3, lr = 1e-3, momentum = 0, lengthscale = None, noise_prior = None):
    # construct likelihood and gp model
    gplh = gpytorch.likelihoods.GaussianLikelihood()

    # lengthscale and noiseprior to return (only applies to RBF kernel, to avoid retraining the noise and lengthscale each iteration on the same train_x)
    lengthscale_return = None; noise_prior_return = None

    if method == "rbf_kernel": 
        gpmodel = ExactGPModel(
            train_x, train_y.squeeze(), gplh, use_rbf_kernel=True
        )
        if torch.is_tensor(lengthscale) and torch.is_tensor(noise_prior):
            gpmodel.likelihood.noise = noise_prior
            gpmodel.covar_module.base_kernel.lengthscale = lengthscale
        else:
            TRAINING_ITER = 100
            # train hyperparams based on training data
            gpmodel.train()
            gplh.train()
            optimizer = torch.optim.Adam(gpmodel.parameters(), lr=0.5)  # includes GaussianLikelihood parameters
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(gplh, gpmodel)
            for i in range(TRAINING_ITER):
                optimizer.zero_grad()
                output = gpmodel(train_x)
                loss = -mll(output, train_y.squeeze())
                loss.backward()
                # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                #     i + 1, TRAINING_ITER, loss.item(),
                #     gpmodel.covar_module.base_kernel.lengthscale,
                #     gpmodel.likelihood.noise
                # ))
                print('Iter %d/%d - Loss: %.3f' % (
                    i + 1, TRAINING_ITER, loss.item()
                ))
                optimizer.step()
            lengthscale_return = gpmodel.covar_module.base_kernel.lengthscale
            noise_prior_return = gpmodel.likelihood.noise
    elif (method == "trained") or (method == "untrained"):
        # generate model
        model = gen_model_specific(n_input_features, n_output_features, nhid)
        if method == "trained":
            utils.train_fullds(model, train_x, train_y.unsqueeze(-1), iterations=iterations, num_batches_per_epoch = num_batches_per_epoch, lr=lr, momentum=momentum)
            # construct likelihood and gp model
            gpmodel = ExactGPModel(
                train_x, train_y, gplh, model, use_linearstrategy = use_linearstrategy
            )
        elif method == "untrained":
            gpmodel = ExactGPModel(
                train_x, train_y, gplh, model, use_linearstrategy = use_linearstrategy
            )
        # set noise to be smaller
        print("residual error: ", torch.mean((model(train_x) - train_y) ** 2))
        with torch.no_grad():
            gplh.noise = torch.max(
                1e-3 * torch.ones(1), torch.mean((model(train_x) - train_y) ** 2)
            )
        print("noise is: ", gplh.noise)
    else:
        raise Exception(f"Method {method} is not recognized.")

    gpmodel.set_train_data(transfer_x, transfer_y, strict=False)
    gpmodel.train()

    # compute the predictive distribution
    with gpytorch.settings.fast_pred_var():
        gpmodel.eval()
        gplh.eval()
        predictive_dist = gpmodel(test_x)
        test_pred = predictive_dist.mean.data
        loss = gplh(gpmodel(test_x)).log_prob(test_y.squeeze())
    
    test_mse = mean_squared_error(test_y, test_pred)

    return loss.detach().numpy(), test_mse, lengthscale_return, noise_prior_return

def get_mse_results_dict(X, Y, N_ADAPTATION_POINTS_LIST,
                        N_INPUT_FEATURES = 11, N_OUTPUT_FEATURES = 1,
                        TRAIN_YEAR = [2004], TEST_YEAR = [2004], HOUR = 8,
                        TRAIN_ZONE_ID = 1, TEST_ZONE_ID = 2, 
                        N_TEST_POINTS = 50, N_HID = 30, N_EPOCHS = 10, BATCHES_PER_EPOCH = 1, LR = 0.01, MOMENTUM = 0,
                        SEED = 1, use_linearstrategy = False):

    train_x, train_y = get_data(X, Y, year_list = TRAIN_YEAR, zone_id = TRAIN_ZONE_ID, hour = HOUR)
    print(f"Shape of train_x: {train_x.shape} train_y: {train_y.shape}")
    _x, _y = get_data(X, Y, year_list = TEST_YEAR, zone_id = TEST_ZONE_ID, hour = HOUR)

    # standardize
    means = train_x.mean(dim=0, keepdim=True)
    stds = train_x.std(dim=0, keepdim=True)
    train_x = (train_x - means) / stds
    _x = (_x - means)/stds

    np.random.seed(SEED)
    idx = np.random.permutation(range(len(_x)))
    test_idx = idx[:N_TEST_POINTS] # always reserve the first N_TEST_POINTS for the test data (will be same across runs, if seed is same)
    test_x, test_y = _x[test_idx], _y[test_idx]

    methods = ['untrained', 'trained', 'rbf_kernel']
    mse_results = {m:[] for m in methods}
    use_lengthscale = None; use_noise_prior = None

    for j, n_adaptation_points in enumerate(N_ADAPTATION_POINTS_LIST):
        assert(n_adaptation_points + N_TEST_POINTS <= len(_x))
        adapt_idx = idx[N_TEST_POINTS:(N_TEST_POINTS + n_adaptation_points)]
        transfer_x, transfer_y = _x[adapt_idx], _y[adapt_idx]
        for method in methods:
            print(method)
            loss, test_mse, lengthscale, noise = get_results(train_x, train_y, transfer_x, transfer_y, test_x, test_y, 
                                        n_input_features = N_INPUT_FEATURES, n_output_features = N_OUTPUT_FEATURES, nhid = N_HID, method = method, 
                                        iterations = N_EPOCHS, num_batches_per_epoch = BATCHES_PER_EPOCH, lr = LR, momentum = MOMENTUM, 
                                        use_linearstrategy = use_linearstrategy, lengthscale = use_lengthscale, noise_prior = use_noise_prior)
            mse_results[method].append(test_mse)
            if torch.is_tensor(lengthscale):
                use_lengthscale = lengthscale
            if torch.is_tensor(noise):
                use_noise_prior = noise
    return mse_results

# Result plotting function
def plot_mse_results(X, Y, mse_results, N_ADAPTATION_POINTS_LIST, title = "", 
                    train_zone = 1, test_zone = 5, train_year = 2004, test_year = 2004, hour = 10):  
    fig, ax = plt.subplots(1,3, figsize = (20,5))   

    ax[0].hist(Y[(Y['zone_id']==train_zone)&(Y['year']==train_year)][f'h{hour}'], color = 'red', alpha = 0.5, label = f'Train');
    ax[0].hist(Y[(Y['zone_id']==test_zone)&(Y['year']==test_year)][f'h{hour}'], color = 'blue', alpha = 0.5, label = f'Test')
    ax[0].legend(loc = 'center right')
    ax[0].set_title(f"Histogram of Electricity Consumption (kW)", fontsize=16)
    ax[0].set_xlabel("Electricity Consumption (kW)", fontsize=16)
    ax[0].set_ylabel("Frequency", fontsize=16)

    ax[1].plot(N_ADAPTATION_POINTS_LIST, mse_results['untrained'], marker = 's', color=sns.color_palette()[0],label='Untrained')
    ax[1].plot(N_ADAPTATION_POINTS_LIST, mse_results['trained'], marker = 's', color=sns.color_palette()[1], label='Trained')
    ax[1].set_xlabel('Number of adaptation points', fontsize=16)
    ax[1].set_ylabel('Test MSE', fontsize=16)
    ax[1].set_yscale('log')
    ax[1].legend(loc = 'best')
    ax[1].set_title("Untrained and trained NTK", fontsize=16)

    ax[2].plot(N_ADAPTATION_POINTS_LIST, mse_results['rbf_kernel'], marker = 's', color=sns.color_palette()[2])
    ax[2].set_xlabel('Number of adaptation points', fontsize=16)
    ax[2].set_ylabel('Test MSE', fontsize=16)
    ax[2].set_yscale('log')
    ax[2].set_title("RBF Kernel", fontsize=16)

    plt.suptitle(title, fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.85])