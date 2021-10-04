from functools import partial
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os


def results_plot(sol, data, depth, time, h, y_total):
    '''
    Not in use anymore
    :param sol:
    :param data:
    :param depth:
    :param time:
    :param h:
    :param y_total:
    :return:
    '''

    plt.rcParams['savefig.dpi'] = 1200
    fig, ax = plt.subplots(1,1)
    res = sol.y[:, -1].reshape((2, int(depth / h)))
    x_pred = np.array([i*h for i in range(int(depth/h))])
    K_pred = res[0, :]
    Na_pred = res[1, :]
    Li_pred = y_total-res[0, :] - res[1, :]
    ax.plot(x_pred,  K_pred, lw=1.5, c='r', label='c(K2O) ')
    ax.plot(x_pred, Na_pred, lw=1.5, c='b', label='c(Na2O)')
    ax.plot(x_pred, Li_pred, lw=1.5, c='m', label='c(Li2O)')
    ax.scatter(data.depth[2:], data.K2O[2:], s=5, c='r', alpha=0.1,  )
    ax.scatter(data.depth[2:], data.Na2O[2:], s=5, c='b', alpha=0.1, )
    ax.scatter(data.depth[2:], data.Li2O[2:], s=5, c='m', alpha=0.1, )
    ax.set_ylim(-0.5, 20.5)
    ax.set_xlabel('Depth (um)', fontsize=13)
    ax.set_ylabel('concentration (mol%)', fontsize=13)
    ax.legend()
    ax.grid(True)
    return ax


class MyTakeStep:

    '''
    Not in use
    For fine controlling of the behaviour of differential evolution method, please check the official
    documentation of differential evolution method in Scipy for details
    '''

    def __init__(self, stepsize=0.5, same_temp=1):
        self.stepsize = stepsize
        self.same_temp = same_temp

    def __call__(self, x, *args, **kwargs):
        s = self.stepsize
        if self.same_temp == 1:
            x[0] += np.random.uniform(-1e-4*s, 1e-4*s)
            x[1:3] += np.random.uniform(-1e-1*s, 1e-1*s, x[1:3].shape)
            x[3:] += np.random.uniform(-s, s, x[3:].shape)
            return x
        elif self.same_temp == 0:
            x[0] += np.random.uniform(-1e-4*s, 1e-4*s)
            x[1:3] += np.random.uniform(-1e-1 * s, 1e-1 * s, x[1:3].shape)
            x[3] += np.random.uniform(-1e-4*s, 1e-4*s)
            x[4:6] += np.random.uniform(-1e-1 * s, 1e-1 * s, x[4:6].shape)
            x[6:] += np.random.uniform(-s, s, x[6:].shape)
            return x


class MyBoundsBasin:

    '''
    Not in use
    For fine controlling of the behaviour of basin hopping method, please check the official
    documentation of basin hopping method in Scipy for details
    '''
    def __init__(self, xmax, xmin, same_temp=1):
        self.same_temp = same_temp
        self.xmax = xmax
        self.xmin = xmin

    def __call__(self, **kwargs):
        x = kwargs['x_new']
        tmax = bool(np.all(x < self.xmax))
        tmin = bool(np.all(x > self.xmin))
        tdiff = True
        t_surf = True
        if self.same_temp == 1:
            tdiff = bool(True)
            t_surf = bool(x[3]/x[4] < x[5]/x[6])
        elif self.same_temp == 0:
            tdiff = bool(np.all(x[0:3] > x[3:6]))
            t_surf = bool(x[6]/x[7] < x[8]/x[9])
        return tmax and tmin and tdiff and t_surf


def con_diff_coeff(x):

    '''
    constraints on diffusion coefficient of two steps' ion exchange
    :param x: x=[Dk_1st, Dna_1st, Dli_1st, Dk_2nd, Dna_2nd, Dli_2nd, Ck_1st, Cna_1st, Ck_2nd, Cna_2nd],
    if the temp of 1st step IOX is higher, then the tracer diffusion coeff shall be larger than those for
    the 2nd step IOX respectively
    :return:
    '''

    return np.array([x[0]-x[3], x[1]-x[4], x[2]-x[5]])


def con_surf_c_st(x):

    '''
    surface concentration constraint when temperatures of both IOX steps are the same
    :param x: x=[Dk, Dna, Dli, Ck_1st, Cna_1st, Ck_2nd, Cna_2nd],
    since the ion exchange temperature is same for both steps, Dk_1st=Dk_2nd=Dk, Dna_1st=Dna_2nd=Dna
    :return: Ck_2nd/Cna_2nd > Ck_1st/Cna_1st
    '''

    return - x[3]/x[4] + x[5]/x[6]


def con_surf_c_1st(x):

    '''
    surface concentration constraint when temperatures of both IOX steps are the same
    :param x: x=[Dk, Dna, Dli, Ck_1st, Cna_1st, Ck_2nd, Cna_2nd],
    :return: 1 > Ck_1st/Cna_1st
    '''

    return x[4]/x[3]-1


def con_surf_c_dt(x):
    '''
    surface concentration constraints when temperatures of both IOX steps are the different
    :param x: x=[Dk_1st, Dna_1st, Dli_1st, Dk_2nd, Dna_2nd, Dli_2nd, Ck_1st, Cna_1st, Ck_2nd, Cna_2nd]
    :return: Ck_2nd/Cna_2nd > Ck_1st/Cna_1st
    '''
    return - x[6]/x[7] + x[8]/x[9]


def print_fun_basin(x, f, accepted, name, method):

    ## LOG info during basin-hopping optimization
    print('for {} by {} : x= {} on minimum = {} accepted_{}'.format(name, method, x, f, int(accepted)))


def print_fun_DE(xk, convergence, name, method):

    ## LOG info during differential evolution optimization

    print('for {} by {} : x= {} on convergence = {} '.format(name, method, xk, convergence))


def shgo_bounds(xmin, xmax):
    '''
    Not in use
    bounds for shgo method
    :param xmin:
    :param xmax:
    :return:
    '''
    bounds = []
    for l, u in zip(xmin, xmax):
        bounds.append((l, u))
    return bounds


def fspro_convertor(file_path, file_name):
    '''
    To convert the output from fspro into a regular excel file
    :param file_path:
    :param file_name:
    :return:
    '''
    df = pd.read_csv(os.path.join(file_path, file_name))
    res = np.zeros((len(df), 2))
    for i in range(len(df)):
        res[i, 0] = float(df.iloc[i, 0].split('\t')[0])
        res[i, 1] = float(df.iloc[i, 0].split('\t')[1])
    cols = ['stress_x', 'stress_y']
    res_df = pd.DataFrame(res, columns=cols)
    res_df.to_csv(os.path.join(file_path, file_name[:-4]+'converted.csv'))

    return res_df


def gen_mesh_grid_csv(x_array, y_array, out_path):
    '''
    Generate mesh grid of chemical toughening condition
    :param x_array:
    :param y_array:
    :param out_path:
    :return:
    '''
    x_m, y_m = np.meshgrid(x_array, y_array)
    time_dict = {'time_1st': pd.Series(x_m.flatten()), 'time_2nd': pd.Series(y_m.flatten())}
    df = pd.DataFrame(time_dict)
    df.to_csv(out_path)
    return df


if __name__ == '__main__':

    # a=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # b = [10, 100, 10e3, 10, 100, 10e3, 20, 20, 20, 20]
    # c=shgo_bounds(a, b)

    ###  to convert the output for FSpro
    # file_path = './data/stress_reg/raw_data/'
    # file_name = 'SLP Daten_0.4_2_b.csv'
    # result = fspro_convertor(file_path, file_name)

    ### to generate time_grid
    x_time = np.arange(120, 360, 5)*60
    y_time = np.arange(60, 300, 5)*60
    out_path = './data/stress_calc/time_grid/700_xup.csv'
    df = gen_mesh_grid_csv(x_time, y_time, out_path)