##### Below are the explanation of stress_fitting_param.txt file ###
'''


1. Each line should end with comma ','. Otherwise error will occur.
2. For each line, please DO NOT change the strings before ':'.


3. Explanation of each line:

{
"D_1st": [3.73E-04, 1.22E+01, 1.87E-01],
# tracer diffusion coefficient for the 1st step IOX, unit: um**2/s

"D_2nd": [3.55E-04, 6.23E+00, 1.64E-01],
# tracer diffusion coefficient for the 2nd step IOX, unit: um**2/s

"c_surf_1st": [2.13E+00, 1.27E+01],
# mole fraction of K, Na at glass surface for the 1st step IOX

"c_surf_2nd": [6.10E+00, 8.72E+00],
# mole fraction of K, Na at glass surface for the 2nd step IOX

"time_1st": 14400,
# time of 1st step ion exchange (unit: second)

"time_2nd": 10800,
# time of 2nd step ion exchange (unit: second)

"c_glass": [0.29, 9.22],
# mole fraction of K, Na in the raw glass

"thickness": 800.0,  # glass thickness (um)


"step_size": 1,  
# unit (um)

"y_total_alkali": 19.37,
# total molar fraction of K+Na+Li in the glass

"E_modulus": 82000,
"poisson_ratio": 0.22

}