import numpy as np
import tensorflow as tf

import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
mpl.style.use('seaborn-paper')  

from src import LorenzPINN

def gen_traindata():
    data = np.load("lorenz_data.npz")
    return data["t"], data["y"]

def lorenz(x, y, z, s=10, r=15, b=2.667):
    '''
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    '''
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot


def fit_lorenz():

    loaded_data = gen_traindata()
    pars = [10,15,8/3]
    lorenz_data = [loaded_data[0], loaded_data[1][:,0], loaded_data[1][:,1], loaded_data[1][:,2]]

    dt = 0.01
    num_steps = 1000
    # Need one more for the initial values
    ts = np.empty((num_steps + 1, 1))
    xs = np.empty((num_steps + 1, 1))
    ys = np.empty((num_steps + 1, 1))
    zs = np.empty((num_steps + 1, 1))

    # Set initial values
    ts[0],xs[0], ys[0], zs[0] = (0., -8, 7, 27)

    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    for i in range(num_steps):
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)
        ts[i + 1] = ts[i] + dt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs[:,0], ys[:,0], zs[:,0], lw=.75)
    ax.set_xlabel("X Axis", fontsize=16)
    ax.set_ylabel("Y Axis", fontsize=16)
    ax.set_zlabel("Z Axis", fontsize=16)
    ax.set_title("Ground-Truth", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=10,pad=2)
    ax.tick_params(axis='both', which='minor', labelsize=8,pad=2)
    plt.savefig('ground_truth.png')
    plt.close()

    lorenz_data = [ts,xs,ys,zs]
    pinn = LorenzPINN(bn=True, log_opt=True, lr=1e-2, layers=3, layer_width=32)
    
    for i in range(6):
        pinn.fit(lorenz_data, pars, 10000, verbose=True)
        curves = pinn.predict_curves(lorenz_data[0])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(curves[0][:,0], curves[1][:,0], curves[2][:,0], lw=.75)
        ax.set_xlabel("X Axis", fontsize=16)
        ax.set_ylabel("Y Axis", fontsize=16)
        ax.set_zlabel("Z Axis", fontsize=16)
        ax.set_title(str((i+1)*10000)+" Epochs", fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=10,pad=2)
        ax.tick_params(axis='both', which='minor', labelsize=8,pad=2)
        plt.savefig(str((i+1)*10000)+" Epochs.png")
        plt.close()

    print(
        "Estimated parameters: c1 = {:3.2f}, c2 = {:3.2f}, c3 = {:3.2f}".format(
            np.exp(pinn.c1.numpy().item()), np.exp(pinn.c2.numpy().item()), np.exp(pinn.c3.numpy().item())
        )  
    )
    print('\n')
    print(
        "True parameters: c1 = {:3.2f}, c2 = {:3.2f}, c3 = {:3.2f}".format(
            pars[0], pars[1], pars[2]
        )
    )                    

if __name__ == "__main__":
    fit_lorenz()