import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
from scipy.integrate import solve_ivp
from timeit import default_timer as timer


def y0_generator(y_surface, y_glass, h, depth):

    '''
    Generate the concentration profile at t=0. the initial condition can be defined use this equation
    :param y_surface: [y_k_surf, y_na_surf], concentration of K, Na at the glass surface, determined by
    the adsorption equilibrium (mol%)
    :param y_glass: [y_k_glass, y_na_glass], composition of K, Na in the raw glass (mol%)
    :param h: step size in depth (um)
    :param depth: thickness/2 (um)
    :return: concentration profile at t=0, first row represents K, second row represents Na
    '''

    y0 = np.zeros([2, int(depth/h)])
    y0[:, 0] = y_surface
    y0[:, -1]= y_glass
    y0[0, 1:-1] = y_glass[0]
    y0[1, 1:-1] = y_glass[1]
    y0_sp = y0.flatten()
    return y0_sp


def diffusion_matrix_nonlinear(D, y, y_total):              # Generalized Doremus's equation
    '''
    NOTE that this function in not in use anymore, Please refer to the second version of this function
    diffusion_matrix_nonlinear_v2

    Calculate the chemical diffusion coefficent from tracer diffusion coefficient using Doremus' equation
    :param D:  tracer diffusion coefficent. [Dk, Dna, Dli]
    :param y: mol fraction of alkali ions at specific depth node [y_K, y_Na]
    :param y_total: total mol fraction of K+Na+Li
    :return: chemical diffusion coefficient
    '''

    temp_sum = y[0]*D[0] + y[1]*D[1] + (y_total-y[0]-y[1])*D[2]
    D_11 = D[0]*(y[1]*D[1]+(y_total-y[1])*D[2])/temp_sum
    D_12 = (D[0]*D[2]  -D[0]*D[1])*y[0]/temp_sum
    D_21 = (D[1]*D[2]-D[0]*D[1])*y[1]/temp_sum
    D_22 = D[1]*(y[0]*D[0]+(y_total-y[0])*D[2])/temp_sum
    return D_11, D_12, D_21, D_22


def diffusion_matrix_nonlinear_v2(D, y, y_total):
    '''
    Calculate the chemical diffusion coefficent from tracer diffusion coefficient using Doremus' equation
    :param D:  tracer diffusion coefficent. [Dk, Dna, Dli]
    :param y: mol fraction of alkali ions at specific depth node [y_K, y_Na]
    :param y_total: total mol fraction of K+Na+Li
    :return: chemical diffusion coefficient
    '''
    temp_sum = y[0, :]*D[0] + y[1, :]*D[1] + (y_total - y[0, :] - y[1, :])*D[2]
    D_11 = D[0]*(y[1, :]*D[1]+(y_total-y[1, :])*D[2])/temp_sum
    D_12 = (D[0]*D[2] - D[0]*D[1])*y[0, :]/temp_sum
    D_21 = (D[1]*D[2]-D[0]*D[1])*y[1, :]/temp_sum
    D_22 = D[1]*(y[0, :]*D[0]+(y_total-y[0, :])*D[2])/temp_sum
    return D_11, D_12, D_21, D_22


def f0(y, time, D, depth, h, y_total, boundary='Neumann'):
    '''
    NOTE that this function in not in use anymore, Please refer to the second version of this function
    f0_v2

    Calculate dC/dt at each depth node, for both K, Na, at certain time node.

    :param y: mol fraction of alkali ions at specific depth node [y_K, y_Na]
    :param time:
    :param D: tracer diffusion coefficient
    :param depth: thickness/2
    :param h: step size for depth
    :param y_total: total mol fraction of K+Na+Li
    :param boundary: type of Boundary condition, 'Dirichlet' or 'Neumann'

    :return: dC_K/dt
    '''
    f = np.zeros(int(depth/h))   # exclude the boundary condition
    # f[0, :] == 0, since the surface concentration is fixed
    depth_node = np.array([i+1 for i in range(len(f)-2)]) # f[0] and f[1] are not updated
    for i in depth_node:
        D_11_f, D_12_f, _, __ = diffusion_matrix_nonlinear(D, y[:, i+1], y_total)
        D_11, D_12,     _, __ = diffusion_matrix_nonlinear(D, y[:, i], y_total)
        D_11_b, D_12_b, _, __ = diffusion_matrix_nonlinear(D, y[:, i-1], y_total)
        f[i] = ((D_11_f+D_11)*(y[0, i+1]-y[0, i])-(D_11+D_11_b)*(y[0, i]-y[0, i-1]))/(2*h**2) + \
               ((D_12_f+D_12)*(y[1, i+1]-y[1, i])-(D_12+D_12_b)*(y[1, i]-y[1, i-1]))/(2*h**2)

    if boundary == 'Dirichlet':
        return f

    elif boundary=='Neumann':
        _, __, D_11_0, D_12_0     = diffusion_matrix_nonlinear(D, y[:, 0], y_total)
        _, __, D_11_0_f, D_12_0_f = diffusion_matrix_nonlinear(D, y[:, 1], y_total)
        _, __, D_11_n, D_12_n     = diffusion_matrix_nonlinear(D, y[:, -1], y_total)
        _, __, D_11_n_b, D_12_n_b = diffusion_matrix_nonlinear(D, y[:, -2], y_total)
        # f[0] = ((D_11_0_f+D_11_0)*(y[0, 1]-y[0, 0])-  (D_11_0+D_11_0_f)*(y[0, 0]-y[0, 1]))/(2*h**2) + \
        #        ((D_12_0_f+D_12_0)*(y[1, 1]-y[1, 0])-  (D_12_0+D_12_0_f)*(y[1, 0]-y[1, 1]))/(2*h**2)
        f[-1]= ((D_11_n_b+D_11_n)*(y[0, -2]-y[0, -1])-(D_11_n+D_11_n_b)*(y[0, -1]-y[0, -2]))/(2*h**2) + \
               ((D_12_n_b+D_12_n)*(y[1, -2]-y[1, -1])-(D_12_n+D_12_n_b)*(y[1, -1]-y[1, -2]))/(2*h**2)

        return f


def f0_v2(y, time, D, depth, h, y_total, boundary='Dirichlet'): # here 0 represents K
    '''


    Calculate dC/dt at each depth node, for both K, Na, at certain time node.

    :param y: mol fraction of alkali ions at specific depth node [y_K, y_Na]
    :param time:
    :param D: tracer diffusion coefficient
    :param depth: thickness/2
    :param h: step size for depth
    :param y_total: total mol fraction of K+Na+Li
    :param boundary: type of Boundary condition, 'Dirichlet' or 'Neumann'

    :return: dC_K/dt
    '''

    f = np.zeros(int(depth / h))  # exclude the boundary condition
    D_11_f, D_12_f, _, __ = diffusion_matrix_nonlinear_v2(D, y[:, 2:], y_total)
    D_11, D_12, _, __ = diffusion_matrix_nonlinear_v2(D, y[:, 1:-1], y_total)
    D_11_b, D_12_b, _, __ = diffusion_matrix_nonlinear_v2(D, y[:, :-2], y_total)
    f[1:-1] = ((D_11_f+D_11)*(y[0, 2:]-y[0, 1:-1])-(D_11+D_11_b)*(y[0, 1:-1]-y[0, :-2]))/(2*h**2) + \
               ((D_12_f+D_12)*(y[1, 2:]-y[1, 1:-1])-(D_12+D_12_b)*(y[1, 1:-1]-y[1, :-2]))/(2*h**2)

    if boundary == 'Dirichlet':
        # print('Na', f[-25:])
        return f

    elif boundary == 'Neumann':
        # _, __, D_11_0, D_12_0     = diffusion_matrix_nonlinear(D, y[:, 0], y_total)
        # _, __, D_11_0_f, D_12_0_f = diffusion_matrix_nonlinear(D, y[:, 1], y_total)
        _, __, D_11_n, D_12_n     = diffusion_matrix_nonlinear(D, y[:, -1], y_total)
        _, __, D_11_n_b, D_12_n_b = diffusion_matrix_nonlinear(D, y[:, -2], y_total)
        # f[0] = ((D_11_0_f+D_11_0)*(y[0, 1]-y[0, 0])-  (D_11_0+D_11_0_f)*(y[0, 0]-y[0, 1]))/(2*h**2) + \
        #        ((D_12_0_f+D_12_0)*(y[1, 1]-y[1, 0])-  (D_12_0+D_12_0_f)*(y[1, 0]-y[1, 1]))/(2*h**2)
        f[-1]= ((D_11_n_b+D_11_n)*(y[0, -2]-y[0, -1])-(D_11_n+D_11_n_b)*(y[0, -1]-y[0, -2]))/(2*h**2) + \
               ((D_12_n_b+D_12_n)*(y[1, -2]-y[1, -1])-(D_12_n+D_12_n_b)*(y[1, -1]-y[1, -2]))/(2*h**2)
        # print('K', f[-25:])
        return f


def f1(y, time, D, depth, h, y_total, boundary='Neumann'):
    '''
    NOTE that this function in not in use anymore, Please refer to the second version of this function
    f1_v2

    Calculate dC/dt at each depth node, for both K, Na, at certain time node.

    :param y: mol fraction of alkali ions at specific depth node [y_K, y_Na]
    :param time:
    :param D: tracer diffusion coefficient
    :param depth: thickness/2
    :param h: step size for depth
    :param y_total: total mol fraction of K+Na+Li
    :param boundary: type of Boundary condition, 'Dirichlet' or 'Neumann'

    :return: dC_Na/dt
    '''


    f = np.zeros(int(depth/h))   # exclude the boundary condition
    depth_node = np.array([i+1 for i in range(len(f)-2)])

    for i in depth_node:
        _, __, D_21_f, D_22_f = diffusion_matrix_nonlinear(D, y[:, i+1], y_total)
        _, __, D_21  , D_22   = diffusion_matrix_nonlinear(D, y[:, i], y_total)
        _, __, D_21_b, D_22_b = diffusion_matrix_nonlinear(D, y[:, i-1], y_total)
        f[i] = ((D_21_f+D_21)*(y[0, i+1]-y[0, i])-(D_21+D_21_b)*(y[0, i]-y[0, i-1]))/(2*h**2) + \
               ((D_22_f+D_22)*(y[1, i+1]-y[1, i])-(D_22+D_22_b)*(y[1, i]-y[1, i-1]))/(2*h**2)

    if boundary == 'Dirichlet':
        return f

    elif boundary=='Neumann':
        _, __, D_21_0, D_22_0     = diffusion_matrix_nonlinear(D, y[:, 0], y_total)
        _, __, D_21_0_f, D_22_0_f = diffusion_matrix_nonlinear(D, y[:, 1], y_total)
        _, __, D_21_n, D_22_n     = diffusion_matrix_nonlinear(D, y[:, -1], y_total)
        _, __, D_21_n_b, D_22_n_b = diffusion_matrix_nonlinear(D, y[:, -2], y_total)
        # f[0] = ((D_21_0_f+D_21_0)*(y[0, 1]-y[0, 0])-(D_21_0+D_21_0_f)*(y[0, 0]-y[0, 1]))/(2*h**2) + \
        #        ((D_22_0_f+D_22_0)*(y[1, 1]-y[1, 0])-(D_22_0+D_22_0_f)*(y[1, 0]-y[1, 1]))/(2*h**2)
        f[-1]= ((D_21_n_b+D_21_n)*(y[0, -2]-y[0, -1])-(D_21_n+D_21_n_b)*(y[0, -1]-y[0, -2]))/(2*h**2) + \
               ((D_22_n_b+D_22_n)*(y[1, -2]-y[1, -1])-(D_22_n+D_22_n_b)*(y[1, -1]-y[1, -2]))/(2*h**2)

        return f


def f1_v2(y, time, D, depth, h, y_total, boundary='Neumann'):

    '''


    Calculate dC/dt at each depth node, for both K, Na, at certain time node.

    :param y: mol fraction of alkali ions at specific depth node [y_K, y_Na]
    :param time:
    :param D: tracer diffusion coefficient
    :param depth: thickness/2
    :param h: step size for depth
    :param y_total: total mol fraction of K+Na+Li
    :param boundary: type of Boundary condition, 'Dirichlet' or 'Neumann'

    :return: dC_Na/dt
    '''

    f = np.zeros(int(depth/h))   # exclude the boundary condition
    _, __, D_21_f, D_22_f = diffusion_matrix_nonlinear_v2(D, y[:, 2:], y_total)
    _, __, D_21, D_22 = diffusion_matrix_nonlinear_v2(D, y[:, 1:-1], y_total)
    _, __, D_21_b, D_22_b = diffusion_matrix_nonlinear_v2(D, y[:, :-2], y_total)
    f[1:-1] = ((D_21_f+D_21)*(y[0, 2:]-y[0, 1:-1])-(D_21+D_21_b)*(y[0, 1:-1]-y[0, :-2]))/(2*h**2) + \
              ((D_22_f+D_22)*(y[1, 2:]-y[1, 1:-1])-(D_22+D_22_b)*(y[1, 1:-1]-y[1, :-2]))/(2*h**2)
    if boundary == 'Dirichlet':
        # print('Na', f[-25:])
        return f

    elif boundary=='Neumann':
        # _, __, D_21_0, D_22_0     = diffusion_matrix_nonlinear(D, y[:, 0], y_total)
        # _, __, D_21_0_f, D_22_0_f = diffusion_matrix_nonlinear(D, y[:, 1], y_total)
        _, __, D_21_n, D_22_n     = diffusion_matrix_nonlinear(D, y[:, -1], y_total)
        _, __, D_21_n_b, D_22_n_b = diffusion_matrix_nonlinear(D, y[:, -2], y_total)
        # f[0] = ((D_21_0_f+D_21_0)*(y[0, 1]-y[0, 0])-(D_21_0+D_21_0_f)*(y[0, 0]-y[0, 1]))/(2*h**2) + \
        #        ((D_22_0_f+D_22_0)*(y[1, 1]-y[1, 0])-(D_22_0+D_22_0_f)*(y[1, 0]-y[1, 1]))/(2*h**2)
        f[-1]= ((D_21_n_b+D_21_n)*(y[0, -2]-y[0, -1])-(D_21_n+D_21_n_b)*(y[0, -1]-y[0, -2]))/(2*h**2) + \
               ((D_22_n_b+D_22_n)*(y[1, -2]-y[1, -1])-(D_22_n+D_22_n_b)*(y[1, -1]-y[1, -2]))/(2*h**2)
        # print('Na', f[-25:])
        return f


def f_sp(t, y_sp, D, depth, h, y_total):

    '''
    NOTE that this function in not in use anymore, Please refer to the second version of this function
    f_sp_v2

    Wrapper function to combine dC_k/dt and dC_na/dt
    :param t: time node (second)
    :param y_sp: concentration profile at t
    :param D: tracer diffusion coefficient
    :param depth: thickness/2
    :param h: step size in depth
    :param y_total: total mol fraction of K+Na+Li
    :return: dC/dt
    '''

    y = y_sp.reshape((2, -1))
    f_0 =f0(y, t, D, depth, h, y_total)
    f_1 =f1(y, t, D, depth, h, y_total)
    f = np.r_[f_0,f_1]
    return f


def f_sp_v2(t, y_sp, D, depth, h, y_total):

    '''


    Wrapper function to combine dC_k/dt and dC_na/dt
    :param t: time node (second)
    :param y_sp: concentration profile at t
    :param D: tracer diffusion coefficient
    :param depth: thickness/2
    :param h: step size in depth
    :param y_total: total mol fraction of K+Na+Li
    :return: dC/dt
    '''

    y = y_sp.reshape((2, -1))
    f_0 =f0_v2(y, t, D, depth, h, y_total)
    f_1 =f1_v2(y, t, D, depth, h, y_total)
    f = np.r_[f_0,f_1]
    return f


def RK4(y, time_node, args=()):


    '''
    NOTE: this function is not in use

    explicit Runge Kutta method, 4th order
    :param y:
    :param time_node:
    :param args:
    :return:
    '''

    k1_0 = f0(y, time_node, *args)
    k2_0 = f0(y + k1_0*t/2, time_node, *args)
    k3_0 = f0(y + k2_0*t/2, time_node, *args)
    k4_0 = f0(y + k3_0*t, time_node, *args)

    k1_1 = f1(y, time_node, *args)
    k2_1 = f1(y + k1_1*t/2, time_node, *args)
    k3_1 = f1(y + k2_1*t/2, time_node, *args)
    k4_1 = f1(y + k3_1*t, time_node, *args)

    y[0, :] = y[0, :] + (t/6)*(k1_0+2*k2_0+2*k3_0+k4_0)
    y[1, :] = y[1, :] + (t/6)*(k1_1+2*k2_1+2*k3_1+k4_1)

    return y


def time_iter(y0, time, t, depth, h, D):
    '''
    NOTE: this function is not in use
    :param y0:
    :param time:
    :param t:
    :param depth:
    :param h:
    :param D:
    :return:
    '''
    y_dense = np.zeros([int(time/t), 2, int(depth/h)])
    y_dense[0, :, :] = y0
    y = y0

    for time_node in range(int(time/t)-1):
        y_update = RK4(y, (time_node+1)*t, args=(D, depth, h))
        y = y_update
        y_dense[time_node+1, :, :] = y_update
        print(y[0, :10])
    return y_dense


def results_plot(y_dense, time, t, depth, h):

    '''
    NOTE: this function is not in use
    :param y_dense:
    :param time:
    :param t:
    :param depth:
    :param h:
    :return:
    '''

    plt.rcParams['savefig.dpi'] = 1200
    fig, ax = plt.subplots(1,1)
    x = np.array([i*h for i in range(int(depth/h))])
    ax.scatter(x, y_dense[-1, 0, :,], s=5, label='K Concentration')
    ax.scatter(x, y_dense[-1, 1, :,], s=5, label='Na Concentration')
    ax.set_xlabel('Depth (um)', fontsize=13)
    ax.set_ylabel('concentration', fontsize=13)
    ax.legend()
    ax.grid(True)

    return fig, ax


def sp_results_plot(y, depth, h):

    '''
    NOTE: this function is not in use

    :param y:
    :param depth:
    :param h:
    :return:
    '''

    plt.rcParams['savefig.dpi'] = 1200
    fig, ax = plt.subplots(1,1)
    x = np.array([i*h for i in range(int(depth/h))])
    ax.scatter(x, y[0, :], s=5, label='K Concentration')
    ax.scatter(x, y[1, :], s=5, label='Na Concentration')
    ax.scatter(x, y_total-y[0, :]-y[1, :], s=5, label='Li Concentration')
    ax.set_xlabel('Depth (um)', fontsize=13)
    ax.set_ylabel('concentration', fontsize=13)
    ax.legend()
    ax.grid(True)

    return fig, ax


def sp_solution(D, y_surface, y_glass, depth, h, time, y_total, iox_step='1st'):
    """

    NOTE that this function in not in use anymore, Please refer to the second version of this function
    sp_solution_v2

    To solve the differential equation group using implicit Euler method

    :param D: tracer diffusion coefficient
    :param y_surface: concentration of K and Na on th glass surface
    :param y_glass: if iox_step=1st, the composition of raw glass, if iox_step=2nd, composition profile
    calculated from 1st step IOX. SHAPE =(1, (depth/h)*2)
    :param depth:
    :param h:
    :param time:
    :param iox_step:
    :return:
    """
    # print('FEM start')
    if iox_step == '1st':
        y0_sp = y0_generator(y_surface, y_glass, h, depth)
    else:
        y_glass = y_glass.reshape((2, int(depth / h)))
        y0_sp = y_glass
        y0_sp[0, 0] = y_surface[0]
        y0_sp[1, 0] = y_surface[1]

    solution = solve_ivp(f_sp, [0, time], y0_sp.flatten(), method='BDF', args=([D, depth, h, y_total]))
    # print('FEM done')
    return solution


def sp_solution_v2(D, y_surface, y_glass, depth, h, time, y_total, iox_step='1st'):
    """

        To solve the differential equation group using implicit Euler method


    :param D: tracer diffusion coefficient
    :param y_surface: concentration of K and Na on th glass surface
    :param y_glass: if iox_step=1st, the composition of raw glass, if iox_step=2nd, composition profile
    calculated from 1st step IOX. SHAPE =(1, (depth/h)*2)
    :param depth: thickness/2
    :param h: step size in depth
    :param time:
    :param iox_step: '1st' or '2nd'

    :return: simulated concentration profiles
    """
    # print('FEM start')
    if iox_step == '1st':
        y0_sp = y0_generator(y_surface, y_glass, h, depth)
    else:
        y_glass = y_glass.reshape((2, int(depth / h)))
        y0_sp = y_glass.copy()   # pay attention!!! otherwise the initial storage will be altered!!
        y0_sp[0, 0] = y_surface[0]
        y0_sp[1, 0] = y_surface[1]
    # t_events = np.array([i for i in range(0, time, 1)])
    solution = solve_ivp(f_sp_v2, [0, time], y0_sp.flatten(), method='BDF',
                         args=([D, depth, h, y_total]))
    # print('FEM done')
    return solution


if __name__ == '__main__':
    '''
    parameters initialization
    0: K, 1:Na, 2: Li
    '''

    D = [27e-5, 4e-2, 2e1]
    depth = 100  # micrometer
    # depth_x = np.r_[np.linspace(0, 10, 50), np.arange(11, depth, 1)]
    time = 60 * 60 * 8 # seconds
    h = 1  # depth step
    t = 1  # time step

    '''
    initial condition and boundary condition initialization
    '''
    y_total = 19.59
    y_surface = np.array([10, 8.5])
    y_glass = np.array([0.339, 9.257])
    # y_dense = time_iter(y0, time, t, depth, h, D)
    # print(y_dense[-1, :, :])
    # results_plot(y_dense, time, t, depth, h)
    y0_sp = y0_generator(y_surface, y_glass, h, depth)
    # sol = solve_ivp(f_sp, [0, time], y0_sp, method='BDF', args=([D, depth, h]))
    # res = sol.y[:,-1].reshape((2, int(depth/h)))
    # print(res)
    # sp_results_plot(res, depth, h)

    # start = timer()
    # sol = sp_solution(D, y_surface, y_glass, depth, h, time, y_total)
    # end = timer()
    # print('time used for old version is {} seconds'.format(end-start))    # c.a. 120 seconds
    # res = sol.y[:, -1].reshape((2, int(depth / h)))
    # f1, a1 = sp_results_plot(res, depth, h)

    start = timer()
    sol = sp_solution_v2(D, y_surface, y_glass, depth, h, time, y_total)
    end = timer()
    print('time used for new version is {} seconds'.format(end-start))   # c.a. 2 seconds
    res = sol.y[:, -1].reshape((2, int(depth / h)))
    f2, a2 = sp_results_plot(res, depth, h)