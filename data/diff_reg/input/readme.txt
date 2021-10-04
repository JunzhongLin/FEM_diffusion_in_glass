##### Below are the explanation of fitting_param.txt file ###
'''


1. Each line should end with comma ','. Otherwise error will occur.
2. For each line, please DO NOT change the strings before ':'.


3. Explanation of each line:

{
"data_name": "input.xlsx",
###  Here, it defined the file containing the edx data

"file_path": "./data/diff_reg/input/",
### this line defined the path of workfolder for the simulation. I suggest not to change it.

"output_name": "xup_800um.pkl",
## the tool will save the results into the pkl file in the same time with a txt file. You can define
## the file name here

"x_ini": [0.0001, 0.25, 0.25, 0.0001, 0.25, 0.25, 2.0, 7.0, 6.0, 3.0],
## the initial guess of the optimization solution, but it will not be used with the differential evolution
## method. x= [Dk_1st, Dna_1st, Dli_1st, Dk_2nd, Dna_2nd, Dli_2nd, y_surface_1st_k, y_surface_1st_na, 
## y_surface_2nd_k, y_surface_2nd_na]

"x_bound_low": [1e-07, 0.001, 0.001, 1e-06, 0.0001, 0.0001, 2, 7, 5, 5],
## lower bound for the solution of x during optimization

"x_bound_high": [0.1, 20.0, 20.0, 0.1, 10.0, 10.0, 10.0, 17.0, 15.0, 10.0],
## higher bound for the solution of x during optimization


"param": [400, [14400, 10800], 1, 16, [0.29, 9.22], 19.37, "X-up_800um"],
## [depth, [time_1st, time_2nd], step, num of points excluded from experimental data, 
##  y_glass: molar fraction of K Na in the glass, y_total: total molar fraction of K Na Li in the glass, 
## name: sample id for verbose]


"same_temp": 0,
## if the temperature of two steps ion exchange are the same, then this value is 1, otherwise 0

"cost_fun_wrapper": 1,
## please do not change. It is related to the cost function used by the differential evolution

"method": "differential_evolution",
## please do not change. I only suggest to use differential evolution.

"ranges": [[1e-06, 0.001], [0.01, 1], [0.01, 1], [1e-06, 0.001], [0.01, 1], [0.01, 1], [0, 15], [0, 15], [0, 15], [0, 15]],
## ranges for the solution of x, which shall match up with the bounds defined earlier

"con_fun_diff": "con_diff_coeff",
"con_fun_surf_st": "con_surf_c_st",
"con_fun_surf_1st": "con_surf_c_1st",
"con_fun_surf_dt": "con_surf_c_dt",
### constraints during the optimization, please check the explanation in the codes for details

"y_total": 19.37,
## total molar fraction of K+Na+Li in the glass

"E_modulus": 82000,
"poisson_ratio": 0.22,
"num_workers": 3
## number of cores of cpu utilized in the calculation

}