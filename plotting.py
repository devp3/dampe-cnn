import matplotlib.pyplot as plt
import numpy as np


def make_pretty_format():
    """Make plots look pretty"""
    import matplotlib
    
    matplotlib.rcParams["font.size"] = 20
    matplotlib.rcParams["xtick.direction"] = "in"
    matplotlib.rcParams["ytick.direction"] = "in"
    matplotlib.rcParams["xtick.major.size"] = 10.
    matplotlib.rcParams["ytick.major.size"] = 10.
    matplotlib.rcParams["xtick.minor.size"] = 8.
    matplotlib.rcParams["ytick.minor.size"] = 8.
    matplotlib.rcParams["legend.fontsize"] = 16
    
def gaus_fit(x, A, mu, sigma):
    return (A / np.sqrt(sigma)) * np.exp(-1.0 * (x - mu)**2 / (2 * sigma**2))
    
def plot_features(features, datasets, labels):
    """
    Plot the input features
    """
    from math import sqrt
    
    num_rows = 1
    num_cols = 1

    if len(features) / sqrt(len(features)) != sqrt(len(features)):
        print("Please provide number of features which are a perfect square")
        return
    
    assert len(datasets)==len(labels), print("Number of datasets and labels do not match")
        
    fig, _ = plt.subplots(int(sqrt(len(features))), int(sqrt(len(features))), figsize=(14,14))
    axes = fig.axes
    
    for i,ax in enumerate(axes):
        feature = features[i]
        for j,dataset in enumerate(datasets):
            ax.hist(dataset[feature], bins=20, alpha=0.5, label=labels[j])
        ax.legend()
        ax.set_title(feature)
    
    return fig

def plot_losses(history):
    """
    Plot the training and validation losses
    """
    fig, ax = plt.subplots(figsize=(16,9))
    
    ax.plot(history.history['loss'], label='Training loss')
    ax.plot(history.history['val_loss'], label='Validation loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend(loc='upper right')
    
    return fig

def plot_predictions(prediction, truth, var_label=['x_top']):
    """
    Plot the predicted vs truth 2D histogram
    
    Inputs:
    prediction: (ndarray) Network predictions for the co-ordinate(s)
    truth: (ndarray) Truth co-ordinate(s)
    var_label: (str) Name of the co-ordinate to be added alongside the truth/predicted label
    
    Outputs:
    Matplotlib figure, axis objects
    """
    
    assert prediction.shape == truth.shape, "Please provide equal-sized arrays"
    
    num_cols, num_rows = 1,1
    fig_size = (12,12)
    
    if len(prediction.shape) > 1:
        num_rows, num_cols = 2,2  #hardcoded for now :(
        fig_size = (40,40)
    
    if len(prediction.shape) == 1:
        prediction = np.reshape(prediction, prediction.shape + (1,))
        truth = np.reshape(truth, truth.shape + (1,))
        var_label = [var_label]
    
    fig, _ = plt.subplots(num_rows, num_cols, figsize=fig_size)
    
    for i_coord, ax in enumerate(fig.axes):
        ax.hist2d(truth[:,i_coord], prediction[:,i_coord], bins=50, cmap='coolwarm')
        
        xmin, xmax = truth[:,i_coord].min()*1.1, truth[:,i_coord].max()*1.1
        ymin, ymax = prediction[:,i_coord].min()*1.1, prediction[:,i_coord].max()*1.1        
        x_values = np.linspace(xmin,xmax,500)
        
        ax.plot(x_values, x_values, linewidth=6, color='white', linestyle='dashed')
        
        ax.set_xlabel(f"True {var_label[i_coord]}")
        ax.set_ylabel(f"Predicted {var_label[i_coord]}")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    
    return fig

def plot_bias(prediction, truth, var_label='x_top', fit_bias=True):
    """
    Plot the bias and fit a gaussian distribution
    
    Inputs:
    prediction: (ndarray) Network predictions for the co-ordinate(s)
    truth: (ndarray) Truth co-ordinate(s)
    var_label: (str) Name of the co-ordinate to be added alongside the bias label
    fit_bias: (bool) Fit the bias using a Gaussian distribution
    
    Outputs:
    Matplotlib figure object
    """
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        print("Cannot load scipy module curve_fit, cannot perform fit")
        fit_bias = False

    assert prediction.shape == truth.shape, "Please provide equal-sized arrays"
    
    num_cols, num_rows = 1,1
    fig_size = (12,12)
    
    if len(prediction.shape) > 1:
        num_rows, num_cols = 2,2
        fig_size = (40,40)
    
    if len(prediction.shape) == 1:
        prediction = np.reshape(prediction, prediction.shape + (1,))
        truth = np.reshape(truth, truth.shape + (1,))
        var_label = [var_label]
    
    bias = prediction - truth
    
    fig, _ = plt.subplots(num_rows, num_cols, figsize=fig_size)
    
    for i_coord, ax in enumerate(fig.axes):
        n, bins, _ = ax.hist(bias[:,i_coord], bins=100, histtype="step", linewidth=2, label="Data")
        ax.axvline(x=0, color='red', linestyle='dashed', linewidth=2)
        
        if fit_bias:
            bin_centers = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
            popt, pcov = curve_fit(gaus_fit, 
                                   xdata=bin_centers, ydata=n, 
                                   p0=[1000., 0., 0.5*(n.max() - n.min())], 
                                   absolute_sigma=False)

            x_values = np.linspace(min(bin_centers), max(bin_centers), 1000)
            ax.plot(x_values, gaus_fit(x_values, *popt), color='tab:orange', linewidth=2, label="Fit")

            ax.text(0.85, 0.8, f"$\mu$ = {popt[1]:0.3f}", 
                     fontsize=18,
                     transform=ax.transAxes)
            ax.text(0.85, 0.75, f"$\sigma$ = {popt[2]:0.3f}", 
                     fontsize=18,
                     transform=ax.transAxes)
            
        y_min = 1e-2 if min(n) >= 0.0 else min(n)*1e-2
        y_max = max(n)*1e2
        ax.set_ylim(y_min, y_max)
        ax.semilogy()
        
        ax.legend(fontsize=16)
        ax.set_xlabel(f"(Pred. - True) {var_label[i_coord]}", fontsize=16)
        ax.set_ylabel(f"Counts", fontsize=16)
        
    return fig

def plot_bias_truth(prediction, truth, var_label='x_top'):
    """
    Plot the bias(es) as a function of truth co-ordinate(s)
    
    Inputs:
    prediction: (ndarray) Network predictions for the co-ordinate(s)
    truth: (ndarray) Truth co-ordinate(s)
    var_label: (str) Name of the co-ordinate to be added alongside the bias label
    
    Outputs:
    Matplotlib figure object
    """
    
    assert prediction.shape == truth.shape, "Please provide equal-sized arrays"
    
    num_cols, num_rows = 1,1
    fig_size = (12,12)
    
    if len(prediction.shape) > 1:
        num_rows, num_cols = 2,2
        fig_size = (40,40)
    
    if len(prediction.shape) == 1:
        prediction = np.reshape(prediction, prediction.shape + (1,))
        truth = np.reshape(truth, truth.shape + (1,))
        var_label = [var_label]
        
    bias = prediction - truth
    
    fig, _ = plt.subplots(num_rows, num_cols, figsize=fig_size)
    
    for i_coord, ax in enumerate(fig.axes):
        ax.scatter(truth[:,i_coord], bias[:,i_coord], marker='o', alpha=0.2, label="Data")
        ax.axhline(y=0, color='red', linestyle='dashed', linewidth=2)
        
        ax.set_xlabel(f"Truth {var_label[i_coord]}", fontsize=16)
        ax.set_ylabel(f"Pred. - Truth", fontsize=16)
        
    return fig
    