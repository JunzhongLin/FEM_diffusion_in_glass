##### Below are the explanation of stress_param.txt file ###
'''


1. Each line should end with comma ','. Otherwise error will occur.
2. For each line, please DO NOT change the strings before ':'.


3. Explanation of each line:

{
"file_path": "./data/stress_calc",
### this line defined the path of workfolder for the simulation. I suggest not to change it. But, if this
### path has been changed, please make sure the new folder includes two files, 'stress_param.txt' and 'input.xlsx'.


"D_1st": [3.73E-04, 1.22E+01, 1.87E-01],
### The default values of diffusion coefficient for the first step IOX are defined here. When the simulation
### starts, they will be overwritten by the values from input.xlsx file for each toughening condition.
### Please note that the unit of diffusion coefficient here is um^2/s, same with the input.xlsx file
### The three values in the list corresponds to the tracer diffusion coefficent of K, Na, Li for the first step
### of ion exchange, [D_K, D_Na, D_Li].


"c_surf_1st": [2.13E+00, 1.27E+01],
### The concentration (mol%) of K, Na on the glass surface for the 1st step IOX are defined here.
### The first value is this list is for K, the second one is for Na.
### The concentration defined here is the default value, it will be overwritten by the values from input.xlsx file
### for each toughening condition.


"time_1st": 14400.0,
### The time of the 1st step IOX is defined here, the unit is seconds.
### The time defined here is the default value, it will be overwritten by the values from input.xlsx file


"temp_1st": 395.0,
### The temperature of the 1st step IOX is defined here, the unit is centigrade.
### The temperature defined here is the default value, it will be overwritten by the values from input.xlsx file
### Please NOTE that this value has no effect on the simulation, it is only the purpose of records.


"D_2nd": [3.55E-04, 6.23E+00, 1.64E-01],
### The default values of diffusion coefficient for the second step IOX are defined here. When the simulation
### starts, they will be overwritten by the values from input.xlsx file for each toughening condition.
### Please note that the unit of diffusion coefficient here is um^2/s, same with the input.xlsx file
### The three values in the list corresponds to the tracer diffusion coefficent of K, Na, Li for the second step
### of ion exchange, [D_K, D_Na, D_Li].


"c_surf_2nd": [6.10E+00, 8.72E+00],
### The concentration (mol%) of K, Na on the glass surface for the 2nd step IOX are defined here.
### The first value is this list is for K, the second one is for Na.
### The concentration defined here is the default value, it will be overwritten by the values from input.xlsx file
### for each toughening condition.


"time_2nd": 10800.0,
### The time of the 2nd step IOX is defined here, the unit is seconds.
### The time defined here is the default value, it will be overwritten by the values from input.xlsx file


"temp_2nd": 380.0,
### The temperature of the 2nd step IOX is defined here, the unit is centigrade.
### The temperature defined here is the default value, it will be overwritten by the values from input.xlsx file
### Please NOTE that this value has no effect on the simulation, it is only the purpose of records.


"thickness": 800,
### The glass thickness is defined here, the unit of it is um.
### ### The thickness defined here is the default value, it will be overwritten by the values from input.xlsx file


"c_glass": [0.29, 9.22],
### The concentration (mol%) of K and Na ions in the un-toughened glass are defined here.
### When the type of glass is changed, the values here in the list should be updated.


"y_total_alkali": 19.37,
### The total concentration (mol%) of K, Na, Li in the un-toughened glass is defined here.
### When the type of glass is changed, the values here in the list should be updated.


"step_size": 1,
### the step_size (the unit is um) in the direction of diffusion depth is defined here.
### The simulation of smaller step_size will take longer computation time.
### Please note that the step_size of time can NOT be defined in this file, because it will be changed
### adaptively during the simulation.


"Young_modulus": 82000,
### The Young's modulus of glass is defined here. The unit is  N/mm^2
### When the type of glass is changed, the values here in the list should be updated.


"Poisson_ratio": 0.22,
### The Poisson ratio of glass is defined here.
### When the type of glass is changed, the values here in the list should be updated.


"dilation_coeff_1st": [0.00183081, 0.00076977],
### the dilation coefficients for the 1st step IOX for K and Na are defined here. The unit is (mol%)^(-1)

"dilation_coeff_2nd": [0.00151281, 0.00080915]
### the dilation coefficients for the 2nd step IOX for K and Na are defined here. The unit is (mol%)^(-1)

}


'''