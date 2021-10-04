import numpy as np
import numerical_method as rk
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, brute, basinhopping, differential_evolution, shgo, \
    NonlinearConstraint, Bounds, LinearConstraint
from scipy import optimize
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import os
import pickle
from functools import partial
from multiprocessing import Pool
from applications import results_plot, MyBoundsBasin, MyTakeStep, print_fun_basin, \
    con_diff_coeff, con_surf_c_dt, con_surf_c_st, con_surf_c_1st, shgo_bounds, print_fun_DE
from timeit import default_timer as timer
import applications


def cost_func_1_step(x, exp_depth, exp_data_K, exp_data_Na, param, same_temp=1,  wrapper=0):
    """
    :param x: [Dk_1st, Dna_1st, Dli_1st, y_surface_1st_k, y_surface_1st_na,]
    :param exp_depth: depth in the edx measurement
    :param exp_data_K: edx profile of K for the IOXed sample
    :param exp_data_Na: edx profile of Na for the IOXed sample
    :param param: [depth, time_1st, step, num]
    :param wrapper: 0 if least square is using, the cost function will return array type data,
                1 if other methods are using, the cost function will return a scalar
    :return: shape = (1, 2*int(depth/h)), differences between pred and exp
    """
    depth, [time_1st, _], h, num, y_glass, y_total, name = param
    D = x[:3]
    y_surface = x[3:]
    sol = rk.sp_solution_v2(D, y_surface, y_glass, depth, h, time_1st, y_total)
    # sol = solve_ivp(rk.f_sp, [0, rk.time], y0_sp, method='BDF', args=(D, ))
    numeric_data = sol.y[:,-1].reshape((2, int(depth/h)))
    numeric_depth = np.array([i * h for i in range(int(depth/h))])
    f_K = interp1d(numeric_depth, numeric_data[0, :])
    f_Na = interp1d(numeric_depth, numeric_data[1, :])
    pred_K = f_K(exp_depth[num:])
    pred_Na = f_Na(exp_depth[num:])

    res =  np.r_[pred_K-exp_data_K[num:], pred_Na-exp_data_Na[num:]]
    if wrapper == 0:
        return res
    elif wrapper == 1:
        return np.sum(res**2)


def cost_func_2_steps(x, data, param, same_temp=1, wrapper=0):
    """
    cost function for the cases which have the edx profiles for both samples after 1st IOX and 2nd IOX

    :param x: [Dk_1st, Dna_1st, Dli_1st, Dk_2nd, Dna_2nd, Dli_2nd,
               y_surface_1st_k, y_surface_1st_na, y_surface_2nd_k, y_surface_2nd_na]
    :param exp_depth_1st: depth in the edx measurement for the 1st Step Ioxed sample
    :param exp_depth_2nd: depth in the edx measurement for the 2nd Step Ioxed sample
    :param exp_data_K_1st: edx profile of K for the 1st Step IOXed sample
    :param exp_data_Na_1st: edx profile of Na for the 1st step IOXed sample
    :param exp_data_K_2nd: edx profile of K for the 2nd step IOXed sample
    :param exp_data_Na_2nd: edx profile of Na for the 2nd step IOXed sample
    :param param: [depth, [time_1st, time_2nd], step, num of points excluded from experimental data,
    y_glass: molar fraction of K Na in the glass, y_total: total molar fraction of K Na Li in the glass,
    name: sample id for verbose]
    :param same_temp: 1 if the temp of two step iox is the same, 0 for different temp
    :param wrapper: 0 if least square is using, the cost function will return array type data,
                    1 if other meothds is using, the cost function will return a scalar
    :return: shape=(1, 4*int(depth/h)), differences between pred and exp
    """
    depth, [time_1st, time_2nd], h, num, y_glass, y_total, name = param

    exp_depth_1st = data.depth_1st.dropna().values
    exp_depth_2nd = data.depth_2nd.dropna().values
    exp_data_K_1st = data.K2O_1st.dropna().values
    exp_data_K_2nd = data.K2O_2nd.dropna().values
    exp_data_Na_1st = data.Na2O_1st.dropna().values
    exp_data_Na_2nd = data.Na2O_2nd.dropna().values

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

    sol_1st = rk.sp_solution_v2(D_1st, y_surface_1st, y_glass, depth, h, time_1st, y_total)
    sol_2nd = rk.sp_solution_v2(D_2nd, y_surface_2nd, sol_1st.y[:, -1], depth, h, time_2nd, y_total,
                                iox_step='2nd')
    numeric_depth = np.array([i * h for i in range(int(depth/h))])
    numeric_data_1st = sol_1st.y[:, -1].reshape((2, int(depth/h)))
    numeric_data_2nd = sol_2nd.y[:, -1].reshape((2, int(depth/h)))
    f_K_1st = interp1d(numeric_depth, numeric_data_1st[0, :])
    f_Na_1st = interp1d(numeric_depth, numeric_data_1st[1, :])
    f_K_2nd = interp1d(numeric_depth, numeric_data_2nd[0, :])
    f_Na_2nd = interp1d(numeric_depth, numeric_data_2nd[1, :])
    pred_K_1st = f_K_1st(exp_depth_1st[num:])
    pred_Na_1st = f_Na_1st(exp_depth_1st[num:])
    pred_K_2nd = f_K_2nd(exp_depth_2nd[num:])
    pred_Na_2nd = f_Na_2nd(exp_depth_2nd[num:])
    res = np.r_[pred_K_1st-exp_data_K_1st[num:], pred_Na_1st-exp_data_Na_1st[num:],
                 pred_K_2nd-exp_data_K_2nd[num:], pred_Na_2nd-exp_data_Na_2nd[num:]]
    if wrapper == 0:
        return res
    elif wrapper == 1:
        return np.sum(res**2)/len(res)


def cost_func_2_steps_simple(x, data, param, same_temp=1, wrapper=0):
    """
    cost function for the cases which only have the edx profile for the sample after 2nd step IOX

    :param x: when same_temp =1, x=[Dk_1st, Dna_1st, Dli_1st, y_surface_1st_k, y_surface_1st_na,
     y_surface_2nd_k, y_surface_2nd_na,]
              when same_temp =0, x=[Dk_1st, Dna_1st, Dli_1st, Dk_2nd, Dna_2nd, Dli_2nd,
               y_surface_1st_k, y_surface_1st_na, y_surface_2nd_k, y_surface_2nd_na]
    :param exp_depth_2nd: depth in the edx measurement for the 2nd Step Ioxed sample
    :param exp_data_K_2nd: edx profile of K for the 2nd step IOXed sample
    :param exp_data_Na_2nd: edx profile of Na for the 2nd step IOXed sample
    :param param: [depth, [time_1st, time_2nd], step, num of points excluded from experimental data,
    y_glass: molar fraction of K Na in the glass, y_total: total molar fraction of K Na Li in the glass,
    name: sample id for verbose]
    :param same_temp: 1 if the temp of two step iox is the same, 0 for different temp
    :param wrapper: 0 if least square is using, the cost function will return array type data,
                    1 if other methods are using, the cost function will return a scalar
                    2 if the user only want to export the FEM solution
    :return: shape = (1, 2*int(depth/h)), differences between pred and exp
    """

    depth, [time_1st, time_2nd], h, num, y_glass, y_total, name = param

    exp_depth_2nd = data.depth_2nd.dropna().values
    exp_data_K_2nd = data.K2O_2nd.dropna().values
    exp_data_Na_2nd = data.Na2O_2nd.dropna().values

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

    # num = 5
    # start = timer()

    sol_1st = rk.sp_solution_v2(D_1st, y_surface_1st, y_glass, depth, h, time_1st, y_total)
    sol_2nd = rk.sp_solution_v2(D_2nd, y_surface_2nd, sol_1st.y[:, -1], depth, h, time_2nd, y_total,
                                iox_step='2nd')
    if wrapper == 2:
        return sol_1st, sol_2nd

    numeric_depth = np.array([i * h for i in range(int(depth / h))])
    numeric_data_2nd = sol_2nd.y[:, -1].reshape((2, int(depth / h)))
    f_K_2nd = interp1d(numeric_depth, numeric_data_2nd[0, :])
    f_Na_2nd = interp1d(numeric_depth, numeric_data_2nd[1, :])
    pred_K_2nd = f_K_2nd(exp_depth_2nd[num:])
    pred_Na_2nd = f_Na_2nd(exp_depth_2nd[num:])

    # end = timer()

    res = np.r_[pred_K_2nd-exp_data_K_2nd[num:], pred_Na_2nd-exp_data_Na_2nd[num:]]
    res_scalar = np.sum(res**2)/len(res)
    # print('{} :__func={}__time={}s \n __x={} '.format(name, res_scalar, end-start, x))
    if wrapper == 0:
        return res
    elif wrapper == 1:
        return res_scalar


def fem_sol_2_step(x, data, param, same_temp=1, wrapper=2):
    _ = data
    sol_1st, sol_2nd = cost_func_2_steps_simple(x, _, _, _, param, same_temp, wrapper)
    return sol_1st, sol_2nd
# def main_regression(cost_fun, data_name, file_path, output_name, x_ini, x_bound_high, x_bound_low,
#                     param, same_temp, method, **kwargs):
#     """
#
#     :param cost_fun: differences between the experiment and simulated edx profile(s)
#     :param data_name: the excel file which stored the experimental edx data
#     :param file_path: the directory where the excel file stored
#     :param output_name:
#     :param x_ini:
#     :param x_bound_high:
#     :param x_bound_low:
#     :param param:
#     :param same_temp:
#     :param method: assign the global minimization method
#     :return:
#     """
#     data_path = os.path.join(file_path, data_name)
#     data = pd.read_excel(data_path)
#     x0 = x_ini
#     res = least_squares(cost_fun, x0, verbose=2, bounds=(x_bound_high, x_bound_low),
#                         args=([data.depth, data.K2O, data.Na2O, param, same_temp]))
#     with open(output_name, 'wb') as handle:
#         pickle.dump(res, handle)
#
#     return res


def main_regression(cost_fun, **kwargs):
    """

    :param cost_fun: differeces between the experiment and simulated edx profile(s)
    :param data_name: the excel file which stored the experimental edx data
    :param file_path: the directory where the excel file stored
    :param output_name:the directory where to save the optimization results (pkl file)
    :param x_ini: the initial guess for the optimization
    :param x_bound_high:
    :param x_bound_low:
    :param param:
    :param same_temp: 1 for same temp of two steps iox, 0 for different temp of two steps iox
    :param cost_fun_wrapper: 0 to return vecotr, 1 to return scalar
    :param method: assign the global minimization method
    :param ranges: ranges for the brutal method
    :return:
    """

    '''variables assignment'''
    data_name = kwargs['data_name']
    file_path = kwargs['file_path']
    output_name = kwargs['output_name']
    x_ini = kwargs['x_ini']
    x_bound_high = kwargs['x_bound_high']
    x_bound_low = kwargs['x_bound_low']
    param = kwargs['param']
    same_temp = kwargs['same_temp']
    cost_fun_wrapper = kwargs['cost_fun_wrapper']
    method = kwargs['method']
    ranges = kwargs['ranges']
    con_fun_diff =     getattr(applications, kwargs['con_fun_diff'])
    con_fun_surf_st =  getattr(applications, kwargs['con_fun_surf_st'])
    con_fun_surf_dt =  getattr(applications, kwargs['con_fun_surf_dt'])
    con_fun_surf_1st = getattr(applications, kwargs['con_fun_surf_1st'])
    num_workers = kwargs['num_workers']
#    con_fun_diff =     kwargs['con_fun_diff']
#    con_fun_surf_st =  kwargs['con_fun_surf_st']
#    con_fun_surf_dt =  kwargs['con_fun_surf_dt']
#    con_fun_surf_1st = kwargs['con_fun_surf_1st']
    y_total = kwargs['y_total']

    data_path = os.path.join(file_path, data_name)
    data = pd.read_excel(data_path)
    x0 = x_ini

    if method == 'least_square':
        res = least_squares(cost_fun, x0, verbose=2, bounds=(x_bound_low, x_bound_high),
                            args=([data, param, same_temp, cost_fun_wrapper]))
        with open(output_name, 'wb') as handle:
            pickle.dump(res, handle)
        return res

    elif method == 'brutal':
        resbrute = brute(cost_fun, ranges, full_output=True, finish=None, disp=True, workers=1, Ns=10,
                         args=([data, param, same_temp, cost_fun_wrapper]))
        with open(method+output_name, 'wb') as handle:
            pickle.dump(resbrute, handle)
        res = least_squares(cost_fun, resbrute[0], verbose=2, bounds=(x_bound_low, x_bound_high),
                            args=([data, param, same_temp, 0]))
        with open('polished'+method+output_name, 'wb') as handle:
            pickle.dump(res, handle)
        return res

    elif method == 'basin_hopping':
        res = basinhopping(cost_fun, x0,
                           minimizer_kwargs={'method': 'L-BFGS-B',
                                             'args': (
                                                 data, param, same_temp, cost_fun_wrapper
                                             ), 'bounds': Bounds(x_bound_low, x_bound_high)},
                           niter=100, take_step=MyTakeStep(same_temp=same_temp), disp=True,
                           accept_test=MyBoundsBasin(xmax=x_bound_high, xmin=x_bound_low, same_temp=same_temp),
                           callback=partial(print_fun_basin, name=data_name, method=method))
        with open(method+output_name, 'wb') as handle:
            pickle.dump(res, handle)
        return res

    elif method == 'differential_evolution':
        # initialize the constraints
        nlc_1 = NonlinearConstraint(con_fun_diff, np.array([0, 0, 0]), np.array([np.inf, np.inf, np.inf]))
        nlc_2 = NonlinearConstraint(con_fun_surf_st, 0, np.inf)
        nlc_3 = NonlinearConstraint(con_fun_surf_dt, 0, np.inf)
        nlc_4 = NonlinearConstraint(con_fun_surf_1st, 0, np.inf)
        A_st = np.array([[0, 0, 0, 1, 1, 0, 0],
                         [0, 0, 0, 0, 0, 1, 1],
                         [0, 0, 0, -1, 0, 1, 0],
                         [0, 0, 0, 0, 1, 0, -1]])
        lc_st = LinearConstraint(A_st, [y_total/2, y_total/2, 0, 0], [y_total, y_total, np.inf, np.inf])
        A_dt = np.array([[0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                         [0, 0, 0, 0, 0, 0, -1, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0, -1]])
        lc_dt = LinearConstraint(A_dt, [y_total/2, y_total/2, 0, 0], [y_total, y_total, np.inf, np.inf])

        if same_temp == 1:
            res = differential_evolution(
                cost_fun, bounds=Bounds(x_bound_low, x_bound_high), maxiter=500, polish=False,
                disp=True, popsize=15, mutation=0.5,
                constraints=(nlc_2, nlc_4, lc_st),
                workers=num_workers, callback=partial(print_fun_DE, name=data_name, method=method),
                args=([data, param, same_temp, cost_fun_wrapper])
                )
        elif same_temp == 0:
            res = differential_evolution(
                cost_fun, bounds=Bounds(x_bound_low, x_bound_high), maxiter=500, polish=False,
                disp=True, popsize=15, mutation=0.5,
                # constraints=(nlc_1, nlc_3, nlc_4, lc_dt),
                constraints=(nlc_1, nlc_3, lc_dt),
                workers=num_workers, callback=partial(print_fun_DE, name=data_name, method=method),
                args=([data, param, same_temp, cost_fun_wrapper])
                )
        # with open(method+output_name, 'wb') as handle:
        #     pickle.dump(res, handle)
        return res

    elif method == 'shgo':
        diff_k = lambda x: con_fun_diff(x)[0]
        diff_na = lambda x: con_fun_diff(x)[1]
        diff_li = lambda x: con_fun_diff(x)[2]
        if same_temp == 1:
            cons =(
                {'type': 'ineq', 'fun': con_fun_surf_st}
            )
            res = shgo(cost_fun, bounds=shgo_bounds(x_bound_low, x_bound_high), constraints=cons,
                       minimizer_kwargs={'method': 'SLSQP',
                                         'args': (
                                             data, param, same_temp, cost_fun_wrapper
                                         ), 'bounds': Bounds(x_bound_low, x_bound_high)},
                       args=([data, param, same_temp, cost_fun_wrapper]))
        elif same_temp == 0:
            cons = (
                {'type': 'ineq', 'fun': diff_k},
                {'type': 'ineq', 'fun': diff_na},
                {'type': 'ineq', 'fun': diff_li},
                {'type': 'ineq', 'fun': con_fun_surf_dt}
            )
            res = shgo(cost_fun, bounds=shgo_bounds(x_bound_low, x_bound_high), constraints=cons,
                       minimizer_kwargs={'method': 'SLSQP',
                                         'args': (
                                             data, param, same_temp, cost_fun_wrapper
                                         ), 'bounds': Bounds(x_bound_low, x_bound_high)},
                       args=([data, param, same_temp, cost_fun_wrapper]))

        return res


def map_main(main_func):
    return main_func()


if __name__ == '__main__':
    '''
    parameter define 
    '''
    path = r'./data/'

    '''
    Global parameters need to be changed
    '''
    y_glass_8797 = np.array([0.21, 2.31])  # y_glass[0]: K2O mol%, y_glass[1]: Na2O mol%
    y_glass_xup = np.array([0.339, 9.257])  # y_glass[0]: K2O mol%, y_glass[1]: Na2O mol%
    y_glass_xup_TPf = np.array([0.29, 9.22])  # y_glass[0]: K2O mol%, y_glass[1]: Na2O mol%
    y_glass_8742 = np.array([0.17, 1.14])  # y_glass[0]: K2O mol%, y_glass[1]: Na2O mol%
    y_total_xup = 19.57
    y_total_8797 = 10.96
    y_total_8742 = 9.99

    param_xup_800um_std = [400, [60 * 60 * 4, 60 * 60 * 3], 1, 5,
                           y_glass_xup, y_total_xup, 'param_xup_800um_std']  # [depth, [time_1st, time_2nd], step, num]
    param_8797_MZ323_2 = [350, [60 * 60 * 16, 60 * 60 * 3], 1, 0, y_glass_8797, y_total_8797,
                          'param_8797_MZ323_2']
    param_8795_2021_TPf = [350, [60 * 60 * 4, 60 * 60 * 2.5], 1, 0, y_glass_xup_TPf, y_total_xup,
                           'param_8795_2021_TPf']
    param_8742_MZ325_2 = [350, [60 * 60 * 16, 60 * 60 * 3], 1, 0, y_glass_8742, y_total_8742,
                          'param_8742_MZ325_2']


    '''
    multiple processing
    '''
    task_1_inputs = {
        'data_name': '8797_MZ323_2.xlsx',
        'file_path': './data',
        'output_name': '8797_MZ323_2.pkl',
        'x_ini': [1.2e-04, 1.5e-01, 5e-01, 2., 7., 6., 3.],
        'x_bound_low': [1e-8, 1e-3, 1e-3, 1., 1.5, 3, 1],
        'x_bound_high': [1e-1, 10., 10., 5., 11., 7.5, 4.5],
        'param': param_8797_MZ323_2,    # param that will be passed to the cost function as args
        'same_temp': 1,
        'cost_fun_wrapper': 1,
        'method': 'differential_evolution',
        'ranges': ((1e-6, 1e-3), (1e-2, 1), (1e-2, 1), (0, 8), (0, 8), (0, 8), (0, 8)),
        'con_fun_diff': con_diff_coeff,
        'con_fun_surf_st': con_surf_c_st,
        'con_fun_surf_1st': con_surf_c_1st,
        'con_fun_surf_dt': con_surf_c_dt,
        'y_total': y_total_8797,
        'num_workers': 3
    }
    task_2_inputs = {
        'data_name': '8795_2021_TPf.xlsx',
        'file_path': './data',
        'output_name': '8795_2021_TPf.pkl',
        'x_ini': [1e-4, 2.5e-1, 2.5e-1, 1e-4, 2.5e-1, 2.5,
                  -1, 2., 7., 6., 3.],
        'x_bound_low': [1e-7, 1e-3, 1e-3, 1e-6, 1e-4, 1e-4, 2, 7, 5, 5],
        'x_bound_high': [1e-1, 20., 20., 1e-1, 10., 10., 10., 17., 15., 10.],
        'param': param_8795_2021_TPf,
        'same_temp': 0,
        'cost_fun_wrapper': 1,
        'method': 'differential_evolution',
        'ranges': ((1e-6, 1e-3), (1e-2, 1), (1e-2, 1), (1e-6, 1e-3), (1e-2, 1), (1e-2, 1),
                   (0, 15), (0, 15), (0, 15), (0, 15)),
        'con_fun_diff': 'con_diff_coeff',
        'con_fun_surf_st': 'con_surf_c_st',
        'con_fun_surf_1st': 'con_surf_c_1st',
        'con_fun_surf_dt': 'con_surf_c_dt',
        'y_total': y_total_xup,
        'num_workers': 3,
        'E_modulus': 82000,                                   #  XUP MegaPascal
        'poisson_ratio': 0.22
    }

    '''
    x_8795=[1.92396089e-04 9.11829052e+00 8.24911049e+00 1.74312480e-04
 7.15297309e+00 3.49124380e-01 2.93487927e+00 8.12516035e+00
 9.82344264e+00 6.84851389e+00] 
    '''
    task_3_inputs = {
        'data_name': '8742_MZ325_2.xlsx',
        'file_path': './data',
        'output_name': '8742_MZ325_2.pkl',
        'x_ini': [1.0e-04, 2.5e-01, 2.5e-01, 2., 7., 6., 3.],
        'x_bound_low': [1e-8, 1e-3, 1e-3, 1, 2, 3.5, 1],
        'x_bound_high': [1e-1, 10., 10., 9., 10., 9., 5.],
        'param': param_8742_MZ325_2,
        'same_temp': 1,
        'cost_fun_wrapper': 1,
        'method': 'differential_evolution',
        'ranges': ((1e-6, 1e-3), (1e-2, 1), (1e-2, 1), (0, 8), (0, 8), (0, 8), (0, 8)),
        'con_fun_diff': con_diff_coeff,
        'con_fun_surf_st': con_surf_c_st,
        'con_fun_surf_1st': con_surf_c_1st,
        'con_fun_surf_dt': con_surf_c_dt,
        'y_total': y_total_8742,
        'num_workers': 3,
    }
    # task_1 = partial(main_regression, cost_func_2_steps_simple, **task_1_inputs)
    # task_2 = partial(main_regression, cost_func_2_steps_simple, **task_2_inputs)
    # task_3 = partial(main_regression, cost_func_2_steps_simple, **task_3_inputs)


    # pool = Pool(3)

    # res_1 = pool.apply_async(task_1)
    # res_2 = pool.apply_async(task_2)
    # res_3 = pool.apply_async(task_3)
    # res_1.get()    # print out the exception if encountered without block
    # res_2.get()
    # res_3.get()
    # results = pool.map(map_main, [task_1, task_2, task_3])
    # pool.close()
    # pool.join()
    method = 'DE'
    output_name = '8797_MZ323_2'
    for i in range(10):
        start = timer()
        res = main_regression(cost_func_2_steps_simple, **task_1_inputs)
        end = timer()
        print('DE_optimization consumes {} minutes'.format((end-start)/60))
        with open(method+output_name+str(i), 'wb') as handle:
            pickle.dump(res, handle)

    '''
    LOAD DATA FROM DISK
    '''

    # file_name = '8797_MZ323_2.xlsx'  ### NEED TO BE CHANGED FOR DIFFERENT SAMPLES
    # data_path = os.path.join(path, file_name)
    # data = pd.read_excel(data_path)
    # y_total = y_total_8797

    '''
    least square fitting to determine the diffusion coefficient and surface 
    concentrations
    '''

    # x0 = [1.2e-04, 1.5e-01, 5e-09, 2, 7, 6, 3]
    # res = least_squares(cost_func_2_steps_simple, x0,
    #                     bounds=([0, 0, 0, 0, 0, 0, 0],
    #                             [10, 100, 10e3, 20, 20, 20, 20]),
    #                     args=([data.depth, data.K2O, data.Na2O, param_8797_MZ323_2, 1]))
    # with open('8797_MZ323_2.pkl', 'wb') as handle:
    #     pickle.dump(res, handle)

    '''
    with known parameters, run the numerical simulation to obtain 
    the concentration profile
    '''
    # with open('batch1_omit_boundary_points.pickle', 'rb') as handle:
    #    res = pickle.load(handle)

    # sol = rk.sp_solution(res.x[:3], res.x[3:], y_glass, depth, h, time)
    # depth, [time_1st, time_2nd], h, num = param_xup_800um_std
    # sol = rk.sp_solution(x0_batch1_corrected[:3], x0_batch1_corrected[3:], y_glass, depth, h, time_1st)
    # ax = results_plot(sol, data, depth, time_1st, h, y_total)

