# . 2D distributions with discriminating lines
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from sklearn import svm

"""
2D example datasets
"""
def get_best_svm_fit(x_1,y_1,x_2,y_2):
    X_1 = [[x_1[i],y_1[i]] for i in range(len(x_1))]
    Y_1 = [[x_2[i],y_2[i]] for i in range(len(x_2))]
    data = X_1 + Y_1
    ans = [0] * len(X_1) + [1] * len(Y_1)
    clf = svm.SVC(kernel='linear')
    clf.fit(data, ans)
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-8, 8)
    yy = a * xx - (clf.intercept_[0]) / w[1]
    return xx, yy

def generate_two_moons_example(x_1,y_1,x_2,y_2):
    X_1 = [[x_1[i],y_1[i]] for i in range(len(x_1)) 
        if (np.sqrt(np.power(x_1[i]-1,2) + np.power(y_1[i],2)) > 2)]
    Y_1 = [[x_2[i],y_2[i]] for i in range(len(x_2)) 
        if (np.sqrt(np.power(x_2[i]+1,2) + np.power(y_2[i],2)) > 2)]
    x_1 = [X_1[i][0] for i in range(len(X_1))]
    y_1 = [X_1[i][1] for i in range(len(X_1))]
    x_2 = [Y_1[i][0] for i in range(len(Y_1))]
    y_2 = [Y_1[i][1] for i in range(len(Y_1))]
    return x_1, y_1, x_2, y_2

def plot_two_moons(x_1,y_1,x_2,y_2,xx,yy):
    fig, axs = plt.subplots(figsize=(10,8))
    axs.scatter(x_1,y_1,color='c',label='moon 1')
    axs.scatter(x_2,y_2,color='m',label='moon 2')
    axs.plot(xx,yy,color='g',linestyle='--',label='SVM')
    axs.set_xlabel("x")
    axs.set_ylabel("y")
    axs.set_title("Two-moons example")
    plt.legend()
    plt.tight_layout()
    plt.show()

def generate_enclosed_blobs_example(x_1,y_1,x_2,y_2):
    X_1 = [[x_1[i],y_1[i]] for i in range(len(x_1)) 
        if (np.sqrt(np.power(x_1[i],2) + np.power(y_1[i],2)) < 1.9)]
    Y_1 = [[x_2[i],y_2[i]] for i in range(len(x_2)) 
        if (np.sqrt(np.power(x_2[i],2) + np.power(y_2[i],2)) > 2.1)]
    x_1 = [X_1[i][0] for i in range(len(X_1))]
    y_1 = [X_1[i][1] for i in range(len(X_1))]
    x_2 = [Y_1[i][0] for i in range(len(Y_1))]
    y_2 = [Y_1[i][1] for i in range(len(Y_1))]
    return x_1, y_1, x_2, y_2

def plot_enclosed_blobs(x_1,y_1,x_2,y_2):
    fig, axs = plt.subplots(figsize=(10,8))
    axs.scatter(x_1,y_1,color='c',label='blob 1')
    axs.scatter(x_2,y_2,color='m',label='blob 2')
    axs.set_xlabel("x")
    axs.set_ylabel("y")
    axs.set_title("Enclosed blobs example")
    plt.legend()
    plt.tight_layout()
    plt.show()

def generate_two_enclosed_blobs_example(x_1,y_1,x_2,y_2):
    X_1 = [[x_1[i],y_1[i]] for i in range(len(x_1)) 
        if (np.sqrt(np.power(x_1[i]+2.5,2) + np.power(y_1[i],2)) < 1.2)]
    Y_1 = [[x_2[i],y_2[i]] for i in range(len(x_2)) 
        if (np.sqrt(np.power(x_2[i]+2.5,2) + np.power(y_2[i],2)) > 1.4 
        and np.sqrt(np.power(x_2[i]-2.5,2) + np.power(y_2[i],2)) > 1.4)]
    Z_1 = [[x_1[i],y_1[i]] for i in range(len(x_1)) 
        if (np.sqrt(np.power(x_1[i]-2.5,2) + np.power(y_1[i],2)) < 1.2)]
    x_1 = [X_1[i][0] for i in range(len(X_1))]
    y_1 = [X_1[i][1] for i in range(len(X_1))]
    x_2 = [Y_1[i][0] for i in range(len(Y_1))]
    y_2 = [Y_1[i][1] for i in range(len(Y_1))]
    x_3 = [Z_1[i][0] for i in range(len(Z_1))]
    y_3 = [Z_1[i][1] for i in range(len(Z_1))]
    return x_1, y_1, x_2, y_2, x_3, y_3

def plot_two_enclosed_blobs(x_1,y_1,x_2,y_2,x_3,y_3):
    fig, axs = plt.subplots(figsize=(10,8))
    axs.scatter(x_1,y_1,color='c',label='blob 1')
    axs.scatter(x_2,y_2,color='m',label='blob 2')
    axs.scatter(x_3,y_3,color='c',label='blob 3')
    axs.set_xlabel("x")
    axs.set_ylabel("y")
    axs.set_title("Several enclosed blobs example")
    plt.legend()
    plt.tight_layout()
    plt.show()

def generate_overlapped_enclosed_blob_example(num_samples=500):
    x_1 = []
    x_2 = []
    while(len(x_1) != num_samples):
        x = np.random.uniform(-6.0,6.0,1)
        y = np.random.uniform(-6.0,6.0,1)
        p = np.random.uniform(0,1.0,1)
        if ((1 / np.sqrt(2*np.pi))*np.exp(-.5*(np.power(x,2.0) + np.power(y,2.0))) > p):
            x_1.append([x,y])

    while(len(x_2) != num_samples):
        x = np.random.uniform(-6.0,6.0,1)
        y = np.random.uniform(-6.0,6.0,1)
        p = np.random.uniform(0,1.0,1)
        if ((1/(2*np.pi))*((1 / np.sqrt(18*np.pi))*np.exp(-(1/18)*(np.power(x,2.0) + np.power(y,2.0))) - (1 / np.sqrt(2.5*np.pi))*np.exp(-(1/2.5)*(np.power(x,2.0) + np.power(y,2.0)))) > p):
            x_2.append([x,y])

    x = [x_1[i][0] for i in range(len(x_1))]
    y = [x_1[i][1] for i in range(len(x_1))]
    x2 = [x_2[i][0] for i in range(len(x_2))]
    y2 = [x_2[i][1] for i in range(len(x_2))]
    return x, y, x2, y2

def plot_overlapped_enclosed_blob(x_1,y_1,x_2,y_2):
    fig, axs = plt.subplots(figsize=(10,8))
    axs.scatter(x_1,y_1,color='c',label='blob 1')
    axs.scatter(x_2,y_2,color='m',label='blob 2')
    axs.set_xlabel("x")
    axs.set_ylabel("y")
    axs.set_title("Overlapped enclosed blobs example")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_simple_network_results(
    signal_in,
    background_in,
    signal_out,
    background_out
):
    plt.subplots(figsize=(15,7.5))
    plt.subplot(1,2,1)
    n, bins, patches = plt.hist( signal_in, 25, density='True', alpha=0.5, facecolor='cyan', label='Signal', hatch="/")
    n2, bins2, patches2 = plt.hist( background_in, 25, density='True', alpha=0.5, facecolor='magenta', label='Background', hatch="/")
    plt.legend( loc='upper right' )
    plt.title( 'Original Signal and Background' )

    plt.subplot(1,2,2)
    plt.hist( signal_out, 25, density='True', alpha=0.5, facecolor='cyan', label='Signal', hatch="/")
    plt.hist( background_out, 25, density='True', alpha=0.5, facecolor='magenta', label='Background', hatch="/")
    plt.legend( loc='upper right' )
    plt.title( 'Network Signal and Background' )
    plt.show()