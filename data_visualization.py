import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
import numerical_method as rk
import c_profile_fitting as cpf
import glob
from mpl_toolkits.mplot3d import Axes3D


def raw_data_plot(file_path, file_name):
    file = os.path.join(file_path, file_name)
    df = pd.read_excel(file)
    fig, ax = plt.subplots(1, 1)
    ax.scatter(df.depth[2:], df.K2O[2:],  s=5, c='r', alpha=0.1, label='c(K2O) ')
    ax.scatter(df.depth[2:], df.Na2O[2:], s=5, c='b', alpha=0.1, label='c(Na2O)')
    ax.scatter(df.depth[2:], df.Li2O[2:], s=5, c='m', alpha=0.1, label='c(Li2O)')
    ax.set_ylim(-0.5, 20.5)
    ax.set_xlabel('Depth (um)', fontsize=13)
    ax.set_ylabel('concentration (mol%)', fontsize=13)
    ax.legend()
    ax.grid(True)
    return fig, ax


def edx_plots(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    fig, ax = plt.subplots(1, 1)
    cols = df.columns[[2, 4, 5, 10]]
    for i, col in enumerate(cols):
        print(i, col)
        ax.scatter(df.iloc[:, 0], df.loc[:, cols[i]], label=cols[i], s=1, alpha=0.5)
    ax2 = ax.twinx()
    ax2.plot(df.iloc[:, 0], df.iloc[:, -1], lw=2, alpha=0.5)
    ax.legend()
    return ax


def check_consistency_L_S(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    fig, ax = plt.subplots(1, 1)
    ax.scatter(df.iloc[:, 0].dropna(), df.iloc[:, 2].dropna(), c='r', alpha=0.4, s=2)
    ax.scatter(df.iloc[:, 3].dropna(), df.iloc[:, 5].dropna(), c='b', alpha=0.4, s=2)
    ax.scatter(df.iloc[:, 3].dropna(), df.iloc[:, 4].dropna(), c='b', alpha=0.4, s=2)
    ax.scatter(df.iloc[:, 0].dropna(), df.iloc[:, 1].dropna(), c='r', alpha=0.4, s=2)
    return ax


def data_plot(raw_data_name, x, param, same_temp=0):
    raw_path = r'./data/diff_reg/raw_data/'
    raw_data_file = os.path.join(raw_path, raw_data_name)
    raw_df = pd.read_excel(raw_data_file)
    if same_temp == 1:
        D_1st = x[:3]
        D_2nd = x[:3]
        y_surface_1st = x[3:5]
        y_surface_2nd = x[5:]
    else:
        D_1st = x[:3]
        D_2nd = x[3:6]
        y_surface_1st = x[6:8]
        y_surface_2nd = x[8:]

    depth, [time_1st, time_2nd], h, num, y_glass, y_total, name = param
    sol_1st = rk.sp_solution_v2(D_1st, y_surface_1st, y_glass, depth, h, time_1st, y_total, iox_step='1st')
    numeric_data_1st = sol_1st.y[:, -1].reshape((2, -1))
    sol_2nd = rk.sp_solution_v2(D_2nd, y_surface_2nd, sol_1st.y[:, -1], depth, h, time_2nd, y_total,
                                iox_step='2nd')
    numeric_depth = np.array([i * h for i in range(int(depth/h))])

    numeric_data_2nd = sol_2nd.y[:, -1].reshape((2, int(depth/h)))

    fig_1, ax_1 = plt.subplots(1, 1)
    ax_1.scatter(raw_df.depth_1st.dropna(), raw_df.K2O_1st.dropna(), s=5, c='r', alpha=0.2, label='c(K2O) ')
    ax_1.scatter(raw_df.depth_1st.dropna(), raw_df.Na2O_1st.dropna(), s=5, c='b', alpha=0.2, label='c(Na2O)')
    ax_1.plot(numeric_depth, numeric_data_1st[0, :], lw=2, c='r', alpha=0.6, label='c(K2O) ')
    ax_1.plot(numeric_depth, numeric_data_1st[1, :], lw=2, c='b', alpha=0.6, label='c(Na2O)')
    ax_1.plot(numeric_depth, y_total-numeric_data_1st[1, :]-numeric_data_1st[0, :], lw=2, c='m',
              alpha=0.6, label='c(Li2O)')
    # ax_1.set_ylim(-0.5, 20.5)
    ax_1.set_xlabel('Depth (um)', fontsize=13)
    ax_1.set_ylabel('concentration (mol%)', fontsize=13)
    ax_1.legend()
    ax_1.grid(True)

    fig_2, ax_2 = plt.subplots(1, 1)
    ax_2.scatter(raw_df.depth_2nd.dropna(), raw_df.K2O_2nd.dropna() , s=5, c='r', alpha=0.2, label='c(K2O) ')
    ax_2.scatter(raw_df.depth_2nd.dropna(), raw_df.Na2O_2nd.dropna(), s=5, c='b', alpha=0.2, label='c(Na2O)')
    # ax_2.scatter(raw_df.depth, raw_df.Li2O, s=5, c='m', alpha=0.1, label='c(Li2O)')
    ax_2.plot(numeric_depth, numeric_data_2nd[0, :], lw=2, c='r', alpha=0.6, label='c(K2O) ')
    ax_2.plot(numeric_depth, numeric_data_2nd[1, :], lw=2, c='b', alpha=0.6, label='c(Na2O)')
    ax_2.plot(numeric_depth, y_total-numeric_data_2nd[1, :]-numeric_data_2nd[0, :], lw=2, c='m',
              alpha=0.6, label='c(Li2O)')
    # ax_2.set_ylim(-0.5, 20.5)
    ax_2.set_xlabel('Depth (um)', fontsize=13)
    ax_2.set_ylabel('concentration (mol%)', fontsize=13)
    ax_2.legend()
    ax_2.grid(True)

    return param, x, ax_1, ax_2


def stress_comparsion(raw_file, calc_file):
    raw_df = pd.read_excel(raw_file)
    calc_df = pd.read_csv(calc_file)
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    cols = raw_df.columns
    for i in range(int(len(raw_df.columns)/4)):
        fig, ax = plt.subplots(1, 1)
        ax.scatter(
            raw_df.iloc[:, 4*i].dropna().values[::10],
            raw_df.iloc[:, [4*i+1, 4*i+2, 4*i+3]].dropna().values[::10].mean(axis=1),
            s=4, alpha=0.5, facecolors='none', edgecolors=color[i],
            label=cols[4*i][:5]+'measured'
        )
        std_stress = raw_df.iloc[:, [4*i+1, 4*i+2, 4*i+3]].dropna().values[::10].std(axis=1)
        ax.fill_between(
            raw_df.iloc[:, 4*i].dropna().values[::10],
            raw_df.iloc[:, [4 * i + 1, 4 * i + 2, 4 * i + 3]].dropna().values[::10].mean(axis=1)-std_stress,
            raw_df.iloc[:, [4 * i + 1, 4 * i + 2, 4 * i + 3]].dropna().values[::10].mean(axis=1)+std_stress,
            alpha=0.3,
            color=color[i]
        )
        ax.plot(
            calc_df.iloc[:, 2*i].dropna().values,
            calc_df.iloc[:, 2*i+1].dropna().values,
            lw=1, alpha=1, c=color[i], label=cols[4*i][:5]+'simulated'
        )
        ax.legend()
        ax.grid(True)
        ax.set_xlabel('Depth, (um)')
        ax.set_ylabel('Compressive stress (MPa)')


def mesh_time_grid_plot(timegrid, raw_file):

    timegrid = pd.read_excel(timegrid)
    raw_df = pd.read_csv(raw_file)

    cs30 = raw_df.iloc[30, :].values[1::2]
    kcs = raw_df.iloc[0, :].values[1::2]
    docl = np.zeros_like(cs30)

    for i in range(int(len(raw_df.columns)/2)):
        docl[i] = np.argmin(np.abs(raw_df.iloc[:, 2*i+1].values))

    x = timegrid.time_1st.values
    y = timegrid.time_2nd.values


    targets = [
        cs30, kcs, docl,
        cs30*docl/cs30.std()/docl.std(),
        cs30*kcs/kcs.std()/cs30.std()
    ]
    titles = ['cs30 (MPa)', 'kcs (MPa)', 'docl (um)', 'cs30*docl', 'cs30*kcs']

    for target, title in zip(targets, titles):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_trisurf(
            timegrid.time_1st / 60.0, timegrid.time_2nd / 60.0, target,
            cmap='viridis', edgecolor='none'
        )
        ax.set_xlabel('time_1st (min)')
        ax.set_ylabel('time_2nd (min)')
        ax.set_zlabel(title)

    return x/60, y/60, targets


def stacked_line_plot(data_file, sheet_name):

    data_df = pd.read_excel(data_file, sheet_name=sheet_name)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    time_1st = np.array([float(i.split('_')[1]) for i in data_df.columns[::2]])
    depth = data_df.iloc[:, 0].dropna().values
    X, Y = np.meshgrid(depth, time_1st)
    Z = np.zeros((len(time_1st), len(depth)))

    for i in range(int(data_df.shape[1]/2)):
        Z[i] = data_df.iloc[:, 2*i+1].dropna().values
        # ax.plot(depth, np.ones_like(depth)*time_1st[i], Z[i], c='g')

    ax.plot_surface(X, Y, Z, rstride=10, cstride=1, color='g', shade=False, lw=10.5)

    ax.set_ylabel('time_1st (min)')
    ax.set_xlabel('depth (um)')

    return ax


if __name__ == '__main__':

    ### plot the comparsion between the simulated and measured stress profile
    raw_file = './data/vis/stress_comparison/SLP_RAW_1st.xlsx'
    calc_file = './data/vis/stress_comparison/output_1st.csv'
    stress_comparsion(raw_file, calc_file)

    # time_grid = './data/vis/3d_plot/700_xup_timegrid.xlsx'
    # raw_file = './data/vis/3d_plot/output.csv'
    # x, y, targets = mesh_time_grid_plot(time_grid, raw_file)
'''
    raw_file = './data/vis/3d_plot/output_2.xlsx'
    ax = stacked_line_plot(raw_file, '2nd_180')

'''



# '''
# initialize the file
# '''
# path = r'./data'
# sample_1 = '8797_MZ323_2.xlsx'
# sample_2 = '8795_2021_TPf.xlsx'
# sample_3 = '8742_MZ325_2.xlsx'
# data_1 = 'DE8797_MZ323_20'
# data_2 = 'differential_evolution8795_2021_TPf.pkl'
# data_3 = 'DE8742_MZ323_20'

# '''
# Global parameters need to be changed
# '''
# y_glass_8797 = np.array([0.21, 2.31])  # y_glass[0]: K2O mol%, y_glass[1]: Na2O mol%
# y_glass_xup = np.array([0.339, 9.257])  # y_glass[0]: K2O mol%, y_glass[1]: Na2O mol%
# y_glass_xup_TPf = np.array([0.29, 9.22])  # y_glass[0]: K2O mol%, y_glass[1]: Na2O mol%
# y_glass_8742 = np.array([0.17, 1.14])  # y_glass[0]: K2O mol%, y_glass[1]: Na2O mol%
# y_total_xup = 19.37
# y_total_8797 = 10.96
# y_total_8742 = 9.99

# param_xup_800um_std = [400, [60 * 60 * 4, 60 * 60 * 3], 1, 5,
#                        y_glass_xup, y_total_xup, 'param_xup_800um_std']  # [depth, [time_1st, time_2nd], step, num]
# param_8797_MZ323_2 = [350, [60 * 60 * 16, 60 * 60 * 3], 0.1, 0, y_glass_8797, y_total_8797,
#                       'param_8797_MZ323_2']
# param_8795_2021_TPf = [400, [60 * 60 * 4, 60 * 60 * 3], 1, 0, y_glass_xup_TPf, y_total_xup,
#                        'param_8795_2021_TPf']
# param_8742_MZ325_2 = [350, [60 * 60 * 16, 60 * 60 * 3], 0.1, 0, y_glass_8742, y_total_8742,
#                       'param_8742_MZ325_2']
# '''
# plot functions
# '''
# test_res_path = r'./DE_result/differential_evolution8795_2021_TPf_2.pkl'

# file = './data/diff_reg/raw_data/consistency_check.xlsx'
# sheet_name = 'test'
# check_consistency_L_S(file, sheet_name)
#   df = pd.read_excel(file_list[0], sheet_name='Sheet1')
#   cols = df.columns[2:]
#   param, x, res, ax_1, ax_2 = data_plot(sample_1, data_1, param_8797_MZ323_2, same_temp=1)
#   ax_2.set_xscale('log', base=10)
#   x=[3.73E-04, 1.22E+01, 1.87E-01, 3.55E-04, 6.23E+00,
#      1.64E-01, 2.13E+00, 1.27E+01, 6.10E+00, 8.72E+00]
#
#   input_file = r'XUP_800_REG_INPUT.xlsx'
#   param, x_in, ax1, ax2 = data_plot(input_file, x, param_8795_2021_TPf)







