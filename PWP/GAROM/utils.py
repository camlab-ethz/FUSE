from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


class ParametricGaussian(object):

    def __init__(self, nx=30, ny=30, domain=[-1, 1], numpy=False) -> None:
        import torch
        import matplotlib
        import matplotlib.tri as tri

        # params
        xx = torch.linspace(domain[0], domain[1], 20)
        yy = xx
        params = torch.cartesian_prod(xx, yy)
        # define domain
        x = torch.linspace(domain[0], domain[1], nx)
        y = torch.linspace(domain[0], domain[1], ny)
        domain = torch.cartesian_prod(x, y)

        self.triang = tri.Triangulation(domain[:, 0], domain[:, 1])

        sol = []
        for p in params:
            sol.append(self.func(domain, p[0], p[1]))
        snapshots = torch.stack(sol)

        if numpy:
            params = params.numpy()
            snapshots = snapshots.numpy()

        self.params = params
        self.snapshots = snapshots

    def func(self, x, mu1, mu2):
        import torch
        x_m1 = (x[:, 0] - mu1).pow(2)
        x_m2 = (x[:, 1] - mu2).pow(2)
        norm = x[:, 0]**2 + x[:, 1]**2
        return torch.exp(-(x_m1 + x_m2))



def preprocessing_lidcavity(key = 'mag(v)', snap_training=None):
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset
    from smithers.dataset import LidCavity

    data = LidCavity()

    keys = data.snapshots.keys()

    if key not in keys:
        raise ValueError(
            f"key not in valid keys, expectd one of {keys} got {key}")
            
    inputs = torch.tensor(data.snapshots[key], dtype=torch.float)
    parameters = torch.tensor(data.params, dtype=torch.float)


    N_snap = inputs.shape[0]
    print(f"N_snap: {N_snap}")

    if snap_training is None:
        snap_training = N_snap

    # splitting indeces for train and test
    indeces_train = np.linspace(0, N_snap-1, num=snap_training, dtype=int)
    mask_train = np.zeros(N_snap, dtype=bool)
    mask_train[indeces_train] = True
    mask_test = np.ones(N_snap, dtype=bool)
    mask_test[indeces_train] = False
    print(f"snap_training: {sum(mask_train)}")
    print(f"snap_testing: {sum(mask_test)}")

    params_training = parameters[mask_train, :]
    snapshots_training = inputs[mask_train, :]
    params_testing = parameters[mask_test, :]
    snapshots_testing = inputs[mask_test, :]

    
    # returning testing and training datasets
    testing_dataset = TensorDataset(
        snapshots_testing, params_testing)
    training_dataset = TensorDataset(
        snapshots_training, params_training)

    return training_dataset, testing_dataset


def preprocessing_gaussian(snap_training=None):
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset

    data = ParametricGaussian()
    inputs = data.snapshots
    parameters = data.params

    N_snap = inputs.shape[0]
    print(f"N_snap: {N_snap}")

    if snap_training is None:
        snap_training = N_snap

    # splitting indeces for train and test
    indeces_train = np.linspace(0, N_snap-1, num=snap_training, dtype=int)
    mask_train = np.zeros(N_snap, dtype=bool)
    mask_train[indeces_train] = True
    mask_test = np.ones(N_snap, dtype=bool)
    mask_test[indeces_train] = False
    print(f"snap_training: {sum(mask_train)}")
    print(f"snap_testing: {sum(mask_test)}")

    params_training = parameters[mask_train, :]
    snapshots_training = inputs[mask_train, :]
    params_testing = parameters[mask_test, :]
    snapshots_testing = inputs[mask_test, :]

    # returning testing and training datasets
    testing_dataset = TensorDataset(
        snapshots_testing, params_testing)
    training_dataset = TensorDataset(
        snapshots_training, params_training)

    return training_dataset, testing_dataset


def preprocessing_graetz(snap_training=None):
    from smithers.dataset import GraetzDataset
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset

    data = GraetzDataset()

    inputs = torch.tensor(data.snapshots, dtype=torch.float)
    parameters = torch.tensor(data.params, dtype=torch.float)

    N_snap = inputs.shape[0]
    print(f"N_snap: {N_snap}")

    if snap_training is None:
        snap_training = N_snap

    # splitting indeces for train and test
    indeces_train = np.linspace(0, N_snap - 1, num=snap_training, dtype=int)
    mask_train = np.zeros(N_snap, dtype=bool)
    mask_train[indeces_train] = True
    mask_test = np.ones(N_snap, dtype=bool)
    mask_test[indeces_train] = False
    print(f"snap_training: {sum(mask_train)}")
    print(f"snap_testing: {sum(mask_test)}")

    # returning testing and training datasets
    testing_dataset = TensorDataset(
        inputs[mask_test, :], parameters[mask_test, :])
    training_dataset = TensorDataset(
        inputs[mask_train, :], parameters[mask_train, :])

    return training_dataset, testing_dataset


def preprocessing_heat(snap_training=None):
    from smithers.dataset import UnsteadyHeatDataset
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset

    data = UnsteadyHeatDataset()

    inputs = torch.tensor(data.snapshots, dtype=torch.float)
    parameters = torch.tensor(data.params, dtype=torch.float)

    N_snap = inputs.shape[0]
    print(f"N_snap: {N_snap}")

    if snap_training is None:
        snap_training = N_snap

    # splitting indeces for train and test
    indeces_train = np.linspace(0, N_snap - 1, num=snap_training, dtype=int)
    mask_train = np.zeros(N_snap, dtype=bool)
    mask_train[indeces_train] = True
    mask_test = np.ones(N_snap, dtype=bool)
    mask_test[indeces_train] = False
    print(f"snap_training: {sum(mask_train)}")
    print(f"snap_testing: {sum(mask_test)}")

    # returning testing and training datasets
    testing_dataset = TensorDataset(
        inputs[mask_test, :], parameters[mask_test, :])
    training_dataset = TensorDataset(
        inputs[mask_train, :], parameters[mask_train, :])

    return training_dataset, testing_dataset


def preprocessing_heat_pod(snap_training=None):
    from smithers.dataset import UnsteadyHeatDataset
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset
    from ezyrb import POD, RBF, Database
    from ezyrb import ReducedOrderModel as ROM

    data = UnsteadyHeatDataset()

    inputs = torch.tensor(data.snapshots, dtype=torch.float)
    parameters = torch.tensor(data.params, dtype=torch.float)

    N_snap = inputs.shape[0]
    print(f"N_snap: {N_snap}")

    if snap_training is None:
        snap_training = N_snap

    # splitting indeces for train and test
    indeces_train = np.linspace(0, N_snap - 1, num=snap_training, dtype=int)
    mask_train = np.zeros(N_snap, dtype=bool)
    mask_train[indeces_train] = True
    mask_test = np.ones(N_snap, dtype=bool)
    mask_test[indeces_train] = False
    print(f"snap_training: {sum(mask_train)}")
    print(f"snap_testing: {sum(mask_test)}")

    # returning testing and training datasets
    testing_dataset = TensorDataset(
        inputs[mask_test, :], parameters[mask_test, :])
    training_dataset = TensorDataset(
        inputs[mask_train, :], parameters[mask_train, :])

    return training_dataset, testing_dataset

def preprocessing_NS(key, snap_training=None, normalize=None):
    from smithers.dataset import NavierStokesDataset
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset

    data = NavierStokesDataset()

    keys = data.snapshots.keys()

    if key not in keys:
        raise ValueError(
            f"key not in valid keys, expectd one of {keys} got {key}")

    inputs = torch.tensor(data.snapshots[key], dtype=torch.float)
    if normalize is not None:
        inputs = normalize(inputs)

    parameters = torch.tensor(data.params, dtype=torch.float)

    N_snap = inputs.shape[0]
    print(f"N_snap: {N_snap}")
    if snap_training is None:
        snap_training = N_snap

    # splitting indeces for train and test
    indeces_train = np.linspace(0, N_snap - 1, num=snap_training, dtype=int)
    mask_train = np.zeros(N_snap, dtype=bool)
    mask_train[indeces_train] = True
    mask_test = np.ones(N_snap, dtype=bool)
    mask_test[indeces_train] = False
    print(f"snap_training: {sum(mask_train)}")
    print(f"snap_testing: {sum(mask_test)}")

    # returning testing and training datasets
    testing_dataset = TensorDataset(
        inputs[mask_test, :], parameters[mask_test, :])
    training_dataset = TensorDataset(
        inputs[mask_train, :], parameters[mask_train, :])

    return training_dataset, testing_dataset

def preprocessing_elastic(key, snap_training=None, normalize=None):
    from smithers.dataset import ElasticBlockDataset
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset

    data = ElasticBlockDataset()

    keys = data.snapshots.keys()

    if key not in keys:
        raise ValueError(
            f"key not in valid keys, expectd one of {keys} got {key}")

    inputs = torch.tensor(data.snapshots[key], dtype=torch.float)
    if normalize is not None:
        inputs = normalize(inputs)

    parameters = torch.tensor(data.params, dtype=torch.float)

    N_snap = inputs.shape[0]
    print(f"N_snap: {N_snap}")
    if snap_training is None:
        snap_training = N_snap

    # splitting indeces for train and test
    indeces_train = np.linspace(0, N_snap - 1, num=snap_training, dtype=int)
    mask_train = np.zeros(N_snap, dtype=bool)
    mask_train[indeces_train] = True
    mask_test = np.ones(N_snap, dtype=bool)
    mask_test[indeces_train] = False
    print(f"snap_training: {sum(mask_train)}")
    print(f"snap_testing: {sum(mask_test)}")

    # returning testing and training datasets
    testing_dataset = TensorDataset(
        inputs[mask_test, :], parameters[mask_test, :])
    training_dataset = TensorDataset(
        inputs[mask_train, :], parameters[mask_train, :])

    return training_dataset, testing_dataset



def assess_model_quality(garom, training_loader, testing_loader,
                         test_name, dataset, plotting=False,
                         printing=False):
    import matplotlib.pyplot as plt
    import torch
    import os
    import matplotlib.ticker as tick
    from math import sqrt
    
    if plotting:
        if not os.path.exists(test_name):
            os.makedirs(test_name)

    garom.to("cpu")
    garom.eval()
    data = dataset()
    error = []
    tot_var = []
    
    for i, dat in enumerate(testing_loader):
        X, valid = dat
        real_dat = X

        # Generate a batch of images
        gen_imgs, variance = garom.estimate(valid, variance=True, mc_steps=100)
        if plotting and i < 15:
            # for generator mean
            fig, axs = plt.subplots(1, 4, figsize=(16, 4), gridspec_kw={'width_ratios': [2, 2, 2, 2]}) 
                                    # constrained_layout=True, 
                                    # subplot_kw={'aspect': 'equal'})
            (ax1, ax2, ax3, ax4) = axs.flat

            pic1 = ax1.tricontourf(data.triang, gen_imgs[0].detach(), levels=120, cmap='viridis')
            cbar1 = plt.colorbar(pic1, ax=ax1, shrink=0.8)
            cbar1.formatter.set_powerlimits((0, 0))
            cbar1.formatter.set_useMathText(True)
            cbar1.ax.yaxis.set_offset_position('left')
            cbar1.ax.tick_params(labelsize=14)

            pic2 = ax2.tricontourf(data.triang, X[0], levels=120, cmap='viridis')  #viridis
            cbar2 = plt.colorbar(pic2, ax=ax2, shrink=0.8)
            cbar2.formatter.set_powerlimits((0, 0))
            cbar2.formatter.set_useMathText(True)
            cbar2.ax.yaxis.set_offset_position('left')
            cbar2.ax.tick_params(labelsize=14)
            ax1.set_title('r-GAROM', fontsize=16)
            ax2.set_title(f'{test_name} original', fontsize=16)

            pic3 = ax3.tricontourf(data.triang, variance[0].detach(), levels=120, cmap='viridis')
            cbar3 = plt.colorbar(pic3, ax=ax3, shrink=0.8)
            cbar3.ax.tick_params(labelsize=14)
            cbar3.formatter.set_powerlimits((0, 0))
            cbar3.formatter.set_useMathText(True)
            cbar3.ax.yaxis.set_offset_position('left')
            pic4 = ax4.tricontourf(data.triang, (X[0]-gen_imgs[0].detach()).pow(2), levels=120, cmap='viridis')
            cbar4 = plt.colorbar(pic4, ax=ax4, shrink=0.8)
            cbar4.formatter.set_powerlimits((0, 0))
            cbar4.formatter.set_useMathText(True)
            cbar4.ax.yaxis.set_offset_position('left')
            cbar4.ax.tick_params(labelsize=14)
            ax3.set_title('r-GAROM variance', fontsize=16)
            ax4.set_title(f'$l_2$ error', fontsize=16)

            # set the aspect ratio
            for ax in axs:
                ax.set_box_aspect(1)
                ax.axis('off')

            # adjust the spacing
            fig.tight_layout(pad=2.0)

            #plt.subplots_adjust(wspace=0.4, hspace=0.5)
            plt.tight_layout()
            plt.savefig(f"{test_name}/testing_gen_{i}.png", dpi=180)
            plt.close()

        norms = torch.linalg.norm(real_dat, dim=-1)
        residuals = torch.linalg.norm(gen_imgs.detach() - real_dat, dim=-1)
        l2_error = (residuals / norms).tolist()
        error += l2_error
        tot_var += variance.mean(dim=-1).tolist()


    final_error_test = 100 * sum(error) / len(error)

    if printing:
        print(
            f" Error on testing: {final_error_test} +- {sqrt(sum(tot_var)/len(tot_var)):.5%}")


    error = []
    tot_var = []
    for i, dat in enumerate(training_loader):
        X, valid = dat
        real_dat = X

        # Generate a batch of images
        gen_imgs, variance = garom.estimate(valid, variance=True)

        norms = torch.linalg.norm(real_dat, dim=-1)
        residuals = torch.linalg.norm(gen_imgs.detach() - real_dat, dim=-1)
        l2_error = (residuals / norms).tolist()
        error += l2_error
        tot_var += variance.mean(dim=-1).tolist()

    final_error_train = 100 * sum(error) / len(error)
    if printing:
        print(
            f" Error on training: {final_error_train} +- {sqrt(sum(tot_var))/len(tot_var):.5%}")

    return final_error_train, final_error_test


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Testing for GAROM, '
                                     'generative adversarial reduced '
                                     'order model.')
    parser.add_argument('--epochs', type=int, default=3000,
                        help='Maximum number of epochs to train GAROM.')
    parser.add_argument('--print_every', type=int, default=None,
                        help='Number of times to print loss during train.')
    parser.add_argument('--gamma', type=float, default=0.75,
                        help='Gamma value for BEGAN training.')
    parser.add_argument('--lambda_k', type=float, default=0.001,
                        help='lambda_kvalue for BEGAN training.')
    parser.add_argument('--encoding_dim', type=int, default=80,
                        help='Dimension to encode data in Discriminator.')
    parser.add_argument('--noise_dim', type=int, default=24,
                        help='Noise dimension for Generator.')
    parser.add_argument('--plot', type=bool, default=False,
                        help='Variable to plot results.')
    parser.add_argument('--print', type=bool, default=False,
                        help='Variable to print results.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed for reproducibility.')
    parser.add_argument('--regular', type=float, default=0.,
                        help='Regularizer.')
    parser.add_argument('--path', type=str, default=None,
                        help='path where data are saved.')
    args = parser.parse_args()
    return args