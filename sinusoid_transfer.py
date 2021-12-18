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

#### GP MODEL CLASS ####

class ExactGPModel(gpytorch.models.ExactGP):
    # exact RBF Gaussian process class
    def __init__(self, train_x, train_y, likelihood, model=None, use_linearstrategy=False, use_rbf_kernel=False):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        if use_rbf_kernel:
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        else:
            self.covar_module = finite_ntk.lazy.NTK(
                model=model, use_linearstrategy=use_linearstrategy
            )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

#### UTILS ####

def gen_reg_task(numdata, seed=None, split=0.0, input_dict=None, extrapolate=False):
    r"""
    gen_reg_task generates data from the same data-generating process as for the 
    sine curves in Bayesian MAML (https://arxiv.org/pdf/1806.03836.pdf). we also
    include a couple of extra options for extrapolation and incorporation of linear
    regression functions (detailed below)
    
    numdata (int): number of data points
    seed (int): for reproducibility
    split (float between 0 and 1): determine the probability of generating a given 
                                linear function
    input_dict (dict - three value): dict for locking down parameters 
                                    (useful for regenerating plots)
    extrapolate (True/False): whether the train data should be U(-5,5) or on a grid
    [-6.5, 6.5] (and we should test extrapolation)
    """

    if not extrapolate:
        train_x = 10.0 * torch.rand(numdata) - 5.0  # U(-5.0, 5.0)
        train_x = torch.sort(train_x)[0]
    else:
        train_x = torch.linspace(-6.5, 6.5, numdata)

    if seed is not None:
        torch.random.manual_seed(seed)

    if torch.rand(1) > split:
        if input_dict is None:
            # same setup as in Bayesian MAML:
            A = 4.9 * torch.rand(1) + 0.1  # should be random on [0.1, 5.0]
            b = 2 * math.pi * torch.rand(1)  # U(0, 2 pi)
            w = 1.5 * torch.rand(1) + 0.5  # U(0.5, 2.0)

            train_y = A * torch.sin(w * (train_x) + b) + (0.01 * A) * torch.randn_like(
                train_x
            )
        else:
            A = input_dict["A"]
            b = input_dict["b"]
            w = input_dict["w"]

            train_y = A * torch.sin(w * (train_x) + b)

    else:
        if input_dict is None:
            A = 6 * torch.rand(1) - 3.0
            b = 6 * torch.rand(1) - 3.0
            w = None

            train_y = A * train_x + b + (0.3) * torch.randn_like(train_x)

        else:
            A = input_dict["A"]
            b = input_dict["b"]
            w = input_dict["w"]
            train_y = A * train_x + b

    return train_x.unsqueeze(-1), train_y.unsqueeze(-1), {"A": A, "b": b, "w": w}


def generate_tasks(n_train_points, n_adaptation_points, n_adaptation_tasks, seed=10, transfer_fn=False):

    torch.random.manual_seed(seed)

    if transfer_fn:
        transfer_split = 0.5
    else:
        transfer_split = 0.0

    task_dataset = []

    # generate the training data
    task_dataset.append(gen_reg_task(n_train_points, split=transfer_split))

    for _ in range(n_adaptation_tasks):
        # generate data
        task_dataset.append(gen_reg_task(n_adaptation_points, split=transfer_split))

    return task_dataset

def gen_model(nhid=40):
    return torch.nn.Sequential(
        torch.nn.Linear(1, nhid),
        torch.nn.Tanh(),
        torch.nn.Linear(nhid, nhid),
        torch.nn.Tanh(),
        torch.nn.Linear(nhid, 1)
    )

def compute_storage_dict(task_dataset, nhid=40, method="untrained"):

    # extract the training data
    train_x, train_y, train_parameters = (
        task_dataset[0][0],
        task_dataset[0][1],
        task_dataset[0][2],
    )

    # generate model
    model = gen_model(nhid)

    # construct likelihood and gp model
    gplh = gpytorch.likelihoods.GaussianLikelihood()
    if method == "rbf_kernel":
        TRAINING_ITER = 30
        gpmodel = ExactGPModel(
            train_x, train_y.squeeze(), gplh, use_rbf_kernel=True
        )
        # train hyperparams based on training data
        gpmodel.train()
        gplh.train()
        optimizer = torch.optim.Adam(gpmodel.parameters(), lr=0.1)  # includes GaussianLikelihood parameters
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(gplh, gpmodel)
        for i in range(TRAINING_ITER):
            optimizer.zero_grad()
            output = gpmodel(train_x)
            loss = -mll(output, train_y.squeeze())
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, TRAINING_ITER, loss.item(),
                gpmodel.covar_module.base_kernel.lengthscale.item(),
                gpmodel.likelihood.noise.item()
            ))
            optimizer.step()
    elif (method == "trained") or (method == "untrained"):
        if method == "trained":
            utils.train_fullds(model, train_x, train_y, iterations=2500, lr=1e-3, momentum=0.9)
            # construct likelihood and gp model
            gpmodel = ExactGPModel(
                train_x, train_y.squeeze(), gplh, model, use_linearstrategy=True
            )
        elif method == "untrained":
            gpmodel = ExactGPModel(
                train_x, train_y.squeeze(), gplh, model, use_linearstrategy=True
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

    plotting_data_dict = {
        "train": {
            "x": train_x.data.numpy(),
            "y": train_y.data.numpy(),
            "pred": model(train_x).data.numpy() if method != "rbf_kernel" else None,
            "true_parameters": train_parameters,
        }
    }

    for task in range(1, len(task_dataset)):

        # Get the transfer (adaptation) points
        transfer_x, transfer_y, task_pars = task_dataset[task]

        # Get the test points (linspace in between adaptation points)
        test_x = torch.linspace(torch.min(transfer_x), torch.max(transfer_x), 1000)
        A = task_pars['A'].numpy()[0]
        b = task_pars['b'].numpy()[0]
        w = task_pars['w'].numpy()[0]
        test_y = A*np.sin(w*test_x + b)

        # here, we reset the training data to ensure that all caches are cleansed
        gpmodel.set_train_data(transfer_x, transfer_y.squeeze(), strict=False)

        start = time.time()
        gpmodel.train()
        #gplh.train()

        # compute the loss
        loss = gplh(gpmodel(transfer_x)).log_prob(transfer_y.squeeze())

        # compute the predictive distribution
        with gpytorch.settings.fast_pred_var():
            gpmodel.eval()
            #gplh.eval()
            interp_x = torch.linspace(
                torch.min(transfer_x) - 2.0, torch.max(transfer_x) + 2.0, 1000
            )
            predictive_dist = gpmodel(interp_x)
            test_pred = gpmodel(test_x).mean.data
            
        pmean = predictive_dist.mean.data
        lower, upper = predictive_dist.confidence_region()
        lower = lower.detach()
        upper = upper.detach()
        test_mse = mean_squared_error(test_y, test_pred)

        end = time.time() - start

        print("Time for task ", task, " is: ", end)

        plotting_data_dict["task" + str(task)] = {
            "x": transfer_x.numpy(),
            "y": transfer_y.numpy(),
            "interp_x": interp_x.numpy(),
            "interp_pred": pmean.numpy(),
            "lower": lower.data.numpy(),
            "upper": upper.data.numpy(),
            "true_parameters": task_pars,
            "loss": loss.detach().numpy(),
            "test_mse": test_mse
        }

    return plotting_data_dict

def plot_storage_dict(storage_dict, method="untrained", print_loss=False, print_test_mse=False):

    # extract the training data and true function
    train_x = storage_dict['train']['x']
    train_y = storage_dict['train']['y']
    nn_pred = storage_dict['train']['pred'] if method != 'rbf_kernel' else None
    
    # get ready for plotting
    nplots = len(storage_dict)-1
    fig, ax = plt.subplots(1, nplots, figsize=(6*nplots, 6))

    # loop through the adaptation tasks
    for i in range(nplots):

        # plot the training data
        ax[i].scatter(train_x, train_y, label='Train data', color=sns.color_palette()[4], s=45)

        # plot the nn predictions
        if method != 'rbf_kernel':
            ax[i].plot(train_x, nn_pred, label='NN Pred', c=sns.color_palette()[2])

        # extract the task specific data
        task_dict = storage_dict[f'task{i+1}']
        transfer_x = task_dict['x']
        transfer_y = task_dict['y']
        interp_x = task_dict['interp_x']
        interp_pred = task_dict['interp_pred']
        lower = task_dict['lower']
        upper = task_dict['upper']
        A = task_dict['true_parameters']['A'].numpy()[0]
        b = task_dict['true_parameters']['b'].numpy()[0]
        w = task_dict['true_parameters']['w'].numpy()[0]
        x_true = np.linspace(-6,6,300)
        y_true = A*np.sin(w*x_true + b)

        # print the Gaussian likelihood loss if specified
        if print_loss:
            loss = task_dict['loss']
            print(f"Gaussian likelihood loss on adaptation task {i+1}: {loss:.4f}")
        # print the test mse if specified
        if print_test_mse:
            test_mse = task_dict['test_mse']
            print(f"Test MSE within range of adaptation data for task {i+1}: {test_mse:.4f}")

        # plot the task specific data
        ax[i].plot(x_true, y_true, label='Adaptation Function', c=sns.color_palette()[1])
        ax[i].scatter(transfer_x, transfer_y, label='Adaptation Data', s=45, marker='v', color=sns.color_palette()[3])
        ax[i].plot(interp_x, interp_pred, label='GP', c=sns.color_palette()[0])
        ax[i].fill_between(interp_x, lower, upper, color=sns.color_palette()[0], alpha=0.1)
        ax[i].set_title(f'Adaptation task {i+1}', fontsize=14)
        if i == 0:
            ax[i].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, ncol=3)

    if method == 'untrained':
        ttl = "GP with un-trained linearized kernel"
    elif method == 'trained':
        ttl = "GP with trained linearized kernel"
    else:
        ttl = "GP with generic RBF kernel"
    fig.suptitle(ttl, fontsize=18)
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])

def get_shifted_sinusoids(n_train_points, n_adaptation_points, A, w, b_list, seed=10):
    '''
    generates training and adaptation data from sinusoids
    of the form A*sin(wx + b) for a range of values of b
    '''

    # set random seed for reproducibility
    torch.random.manual_seed(seed)

    # convert inputs to tensors
    A = torch.tensor([A])
    w = torch.tensor([w])

    task_dataset = []

    for i, b in enumerate(b_list):

        b = torch.tensor([b])

        if i == 0:
            train_x = 10.0 * torch.rand(n_train_points)
        else:
            train_x = 10.0 * torch.rand(n_adaptation_points)
        train_y = A*torch.sin(train_x*w + b)

        task_dataset.append((train_x.unsqueeze(-1), train_y.unsqueeze(-1), {"A": A, "b": b, "w": w}))

    return task_dataset

def get_mse_shifted_sinusoids(n_train_points, n_adaptation_points_list, A, w, b_list):

    # dictionary to hold test mses for each method
    methods = ['untrained', 'trained', 'rbf_kernel']
    test_mses = {method: {f'task{i}': [] for i in range(1, len(b_list))} for method in methods}
    untrained_test_mses = []
    trained_test_mses = []
    rbf_kernel_test_mses = []
    
    # loop through list of adaptation points
    for j, n_adaptation_points in enumerate(n_adaptation_points_list):

        # generate data
        task_dataset = get_shifted_sinusoids(n_train_points=n_train_points, n_adaptation_points=n_adaptation_points, A=A, w=w, b_list=b_list)

        # loop through methods
        for method in methods:

            # compute the storage dictionary
            storage_dict = compute_storage_dict(task_dataset, nhid=40, method=method)

            # loop through adaptation tasks
            for i in range(1, len(b_list)):

                # update test mse dict
                task_dict = storage_dict[f'task{i}']
                task_mse = task_dict['test_mse']
                mse_list = test_mses[method][f'task{i}']
                mse_list.append(task_mse)
                test_mses[method][f'task{i}'] = mse_list

    return test_mses, storage_dict

def plot_mse_shifted_sinusoids(test_mses, storage_dict, n_adaptation_points_list, A, w, b_list, savefig=False):

    # list out method
    methods = ['untrained', 'trained', 'rbf_kernel']

    # initialize a figure
    fig, ax = plt.subplots(len(b_list)-1, 2, figsize=(14, 5*(len(b_list)-1)))

    # plot train and adaptation data
    for i in range(1, len(b_list)):
        train_x = storage_dict['train']['x']
        train_y = storage_dict['train']['y']
        A_train = storage_dict['train']['true_parameters']['A'].numpy()[0]
        b_train = storage_dict['train']['true_parameters']['b'].numpy()[0]
        w_train = storage_dict['train']['true_parameters']['w'].numpy()[0]
        x_true = np.linspace(0,10,300)
        y_true = A_train*np.sin(w_train*x_true + b_train)
        A_adapt = storage_dict[f'task{i}']['true_parameters']['A'].numpy()[0]
        b_adapt = storage_dict[f'task{i}']['true_parameters']['b'].numpy()[0]
        w_adapt = storage_dict[f'task{i}']['true_parameters']['w'].numpy()[0]
        y_adapt = A_adapt*np.sin(w_adapt*x_true + b_adapt)
        ax[i-1][1].scatter(train_x, train_y, label='train points', s=45, color=sns.color_palette()[4])
        ax[i-1][1].plot(x_true, y_true, label='train function', c=sns.color_palette()[3])
        ax[i-1][1].plot(x_true, y_adapt, label='adaptation function', c=sns.color_palette()[7])
        if i == len(b_list)-1:
            ax[i-1][1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), fancybox=True, ncol=3, fontsize=14)

    # plot test MSEs
    for i in range(1, len(b_list)):
        for j, method in enumerate(methods):
            mse_list = test_mses[method][f'task{i}']
            ax[i-1][0].plot(n_adaptation_points_list, mse_list, color=sns.color_palette()[j], marker='s', label=method)
        ax[i-1][0].set_xlabel('Number of adaptation points', fontsize=16)
        ax[i-1][0].set_ylabel('Test MSE', fontsize=16)
        ax[i-1][0].set_yscale('log')
        ax[i-1][0].set_title(f'Adaptation Task {i}', fontsize=18)
        if i == len(b_list)-1:
            ax[i-1][0].legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), fancybox=True, ncol=3, fontsize=14)
    fig.tight_layout()
    if savefig:
        plt.savefig('shifted_sinusoids.png', dpi=300, bbox_inches='tight')