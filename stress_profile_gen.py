import numpy as np
import numerical_method as rk
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import os
import pickle
from scipy.integrate import quad, cumtrapz
from scipy.interpolate import interp1d


def k_func(kcs, dol, x):
    return -kcs/dol*x + kcs


def measured_data_gen(kcs, k_dol, slp_file_path, depth):
    slp_data = pd.read_excel(slp_file_path)
    x = slp_data.iloc[:,0].dropna().values
    k_y = k_func(kcs, k_dol, x)
    na_y = slp_data.iloc[:, 1].dropna().values
    y = np.r_[k_y[k_y > na_y], na_y[k_y <= na_y]]
    x = x[x<depth-1]
    y = y[:len(x)]

    return x, y


def c_profile_gen(d_file, y_glass, depth, h, time):
    with open(d_file, 'rb') as handle:
        res = pickle.load(handle)
    sol = rk.sp_solution(res.x[:3], res.x[3:], y_glass, depth, h, time)
    x = np.array([i * h for i in range(int(depth / h))])
    c = sol.y[:, -1].reshape((2, int(depth / h)))   ## c_k = c[0, :], c_na = c[1, :]
    return x, c


def c_profile_gen_2_steps(param):
    '''
    :param param:
    :return:
    '''
    h = param['step_size']

    sol_1st = rk.sp_solution_v2(
        param['D_1st'], param['c_surf_1st'], param['c_glass'],
        param['thickness']/2, param['step_size'], param['time_1st'],
        param['y_total_alkali']
    )
    sol_2nd = rk.sp_solution_v2(
        param['D_2nd'], param['c_surf_2nd'], sol_1st.y[:, -1],
        param['thickness']/2, param['step_size'], param['time_2nd'],
        param['y_total_alkali'], iox_step='2nd'
    )
    depth = np.array([i * h for i in range(int(param['thickness'] / 2 / h))])
    c_1st = sol_1st.y[:, -1].reshape((2, int(param['thickness'] / 2 / h)))
    c_2nd = sol_2nd.y[:, -1].reshape((2, int(param['thickness'] / 2 / h)))

    return depth, c_1st, c_2nd


def numeric_stress_profile_gen(E, v, b_k, b_na, c_profile, depth, h):
    x = np.array([i * h for i in range(int(depth / h))])
    c_k_avg = cumtrapz(c_profile[0, :], x, initial=0)[-1]/(depth-1)
    c_na_avg = cumtrapz(c_profile[1, :], x, initial=0)[-1]/(depth-1)
    stress = E*b_k*(c_profile[0, :]-c_k_avg)/(1-v) + E*b_na*(c_profile[1, :]-c_na_avg)/(1-v)
    return x, stress


def cost_function_stress(opt_x, x_measured, stress_measured, depth, h, E, v, c_profile):
    b_k, b_na = opt_x
    x_cal, stress_cal = numeric_stress_profile_gen(E, v, b_k, b_na, c_profile, depth, h)
    f_stress = interp1d(x_cal, stress_cal)
    return stress_measured-f_stress(x_measured)


def dilation_coeff_reg(opt_x_ini, params, measured_stress_profile, IOX_step='2nd'):

    h = params['step_size']
    E = params['E_modulus']
    v = params['poisson_ratio']
    x_measured, stress_measured = measured_stress_profile
    # x_measured, stress_measured = measured_data_gen(kcs, k_dol, slp_file, depth)
    _, c_1st, c_2nd = c_profile_gen_2_steps(params)
    depth = params['thickness']/2

    if IOX_step == '1st':

        args_cost_func = (x_measured, stress_measured, depth, h, E, v, c_1st)
        res = least_squares(cost_function_stress, opt_x_ini, args=args_cost_func)
        x_cal, stress_cal = numeric_stress_profile_gen(E, v, *res.x, c_1st, depth, h)

    else:
        args_cost_func = (x_measured, stress_measured, depth, h, E, v, c_2nd)
        res = least_squares(cost_function_stress, opt_x_ini, args=args_cost_func)
        x_cal, stress_cal = numeric_stress_profile_gen(E, v, *res.x, c_2nd, depth, h)

    return res, x_cal, stress_cal


def stress_plot(x_c, s_c, x_m, s_m):
    plt.rcParams['savefig.dpi'] = 1200
    fig, ax = plt.subplots(1,1)

    ax.plot(x_m, s_m, c='r', label='measured stress')
    ax.plot(x_c, s_c, c='g', label='simulated stress', ls='--', alpha=0.8)
    ax.legend()
    ax.grid(True)
    ax.set_ylim(-120, 1100)
    ax.set_ylabel('compressive stress (MPa)')
    ax.set_xlabel('depth (um)')

    return fig, ax

if __name__ == '__main__':
    '''
        path = r'./data/slp'
    slp_filename = r'batch_1.xlsx'
    slp_file = os.path.join(path, slp_filename)
    parameter_file = r'batch1_omit_boundary_points.pickle_num_1.pickle'
    depth = 800 / 2
    time = 60 * 60 * 4
    h = 0.1
    E_modulus_xup = 82000  # megapascal
    poisson_ratio_xup = 0.22
    kcs = 430 / 0.813
    k_dol = 3.8
    y_glass = np.array([0.339, 9.257])

    x_measure, stress_measure = measured_data_gen(kcs, k_dol, slp_file, depth)
    params = (kcs, k_dol, depth, h, time, E_modulus_xup, poisson_ratio_xup, y_glass)
    results, x_cal, stress_cal = dilation_coeff_reg((200, 10), params)


    stress_plot(x_cal, stress_cal, x_measure, stress_measure)
    '''
    pass
