import os 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def create_plot(plot_name, plot_title, csv_file):
    data = np.genfromtxt(csv_file, delimiter=',', skip_header=1, names=['time', 'step', 'value'])
    #Smooth data
    window_size = 5
    data['value'] = pd.Series(data['value']).rolling(window_size, min_periods=window_size).mean()
    #Plot
    plt.figure()
    plt.title(plot_title)
    plt.plot(data['step'], data['value'])
    plt.xlabel("Minibatches")
    plt.legend()
    plt.savefig(plot_name, dpi=100)

def plot_kl(plot_dir, csv_dir):
    plot_name = os.path.join(plot_dir, "KL_divergence.png")
    csv_file = os.path.join(csv_dir, "10_logistic_kl_div.csv")
    plot_title = "Training - KL divergence"
    create_plot(plot_name, plot_title, csv_file)

def plot_action_loss(plot_dir, csv_dir):
    plot_name = os.path.join(plot_dir, "Action_loss.png")
    csv_file = os.path.join(csv_dir, "10_logistic_action_loss.csv")
    plot_title = "Training - Action Likelihood Loss"
    create_plot(plot_name, plot_title, csv_file)

def plot_total_loss(plot_dir, csv_dir):
    plot_name = os.path.join(plot_dir, "Total_loss.png")
    csv_file = os.path.join(csv_dir, "10_logistic_total_loss.csv")
    plot_title = "Training - Total Loss"
    create_plot(plot_name, plot_title, csv_file)

def plot_val_accuracy(plot_dir, csv_dir):
    plot_name = os.path.join(plot_dir, "Val_accuracy.png")
    csv_file = os.path.join(csv_dir, "10_logistic_val_accuracy.csv")
    plot_title = "Validation - Action Accuracy"
    create_plot(plot_name, plot_title, csv_file)

def plot_val_action_loss(plot_dir, csv_dir):
    plot_name = os.path.join(plot_dir, "Val_action_loss.png")
    csv_file = os.path.join(csv_dir, "10_logistic_val_action_loss.csv")
    plot_title = "Validation - Action Likelihood Loss"
    create_plot(plot_name, plot_title, csv_file)

dir_path = os.path.dirname(os.path.realpath(__file__))
csv_dir = os.path.join(dir_path, "analysis/csv")
plot_dir = os.path.join(dir_path, "analysis/plots")  

if os.path.exists(csv_dir):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_kl(plot_dir, csv_dir)
    plot_action_loss(plot_dir, csv_dir)
    plot_total_loss(plot_dir, csv_dir)
    plot_val_accuracy(plot_dir, csv_dir)
    plot_val_action_loss(plot_dir, csv_dir)