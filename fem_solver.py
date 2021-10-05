import argparse
import json
import pandas as pd
from stress_profile_gen import c_profile_gen_2_steps, numeric_stress_profile_gen
from logconf import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class FemSolver:
    def __init__(self, sys_argv=None):
        parser = argparse.ArgumentParser()
        parser.add_argument('--input_file', default='./data/stress_calc/input.xlsx')
        parser.add_argument('--stress_calc_param_path', default='./data/stress_calc/stress_param.txt')
        parser.add_argument('--repeated_times', type=int, default=10, help='define the times of regression')
        parser.add_argument('--output_file', default='./data/stress_calc/output.csv')
        parser.add_argument('--comments', type=str, default='none', help='add comments to describe the IOX')

        self.stress_args = parser.parse_args('')
        self.stress_params = self._init_params()
        self.toughening_params = self._init_toughening_param()
        assert 1-any(i=='_' for i in self.stress_args.comments), 'illegal symbol ''_'' used in the comments'

    def _init_params(self):
        with open(self.stress_args.stress_calc_param_path) as f:
            stress_params = json.load(f)
        return stress_params

    def _init_toughening_param(self):
        toughening_param = pd.read_excel(self.stress_args.input_file)
        return toughening_param

    def main(self):
        stress_dict = {}
        for i in range(len(self.toughening_params)):
            log.info('Simulation on going at {}/{}'.format(i+1, len(self.toughening_params)))
            temp_params = self.stress_params
            temp_params['thickness'] = self.toughening_params.iloc[i].thickness
            temp_params['time_1st'] = self.toughening_params.iloc[i].time_1st
            temp_params['temp_1st'] = self.toughening_params.iloc[i].temp_1st
            temp_params['time_2nd'] = self.toughening_params.iloc[i].time_2nd
            temp_params['temp_2nd'] = self.toughening_params.iloc[i].temp_2nd
            temp_params['D_1st'] = json.loads(self.toughening_params.iloc[i].D_1st)
            temp_params['D_2nd'] = json.loads(self.toughening_params.iloc[i].D_2nd)
            temp_params['c_surf_1st'] = json.loads(self.toughening_params.iloc[i].c_surf_1st)
            temp_params['c_surf_2nd'] = json.loads(self.toughening_params.iloc[i].c_surf_2nd)

            depth, c_1st, c_2nd = c_profile_gen_2_steps(temp_params)

            stress_x_1st, stress_y_1st = numeric_stress_profile_gen(
                temp_params['Young_modulus'], temp_params['Poisson_ratio'], temp_params['dilation_coeff_1st'][0],
                temp_params['dilation_coeff_1st'][1], c_1st, temp_params['thickness']/2, temp_params['step_size']
            )
            stress_x_2nd, stress_y_2nd = numeric_stress_profile_gen(
                temp_params['Young_modulus'], temp_params['Poisson_ratio'], temp_params['dilation_coeff_2nd'][0],
                temp_params['dilation_coeff_2nd'][1], c_2nd, temp_params['thickness'] / 2, temp_params['step_size']
            )

            stress_dict[
                str(temp_params['thickness'] / 1000) + '_'
                + str(temp_params['time_1st'] / 60) + '_stress_x'
                ] = pd.Series(stress_x_1st)

            stress_dict[
                str(temp_params['thickness'] / 1000) + '_'
                + str(temp_params['time_1st'] / 60) + '_stress_y'
                ] = pd.Series(stress_y_1st)

            stress_dict[
                str(temp_params['thickness']/1000) + '_'
                + str(temp_params['time_1st']/60) + '_'
                + str(temp_params['time_2nd']/60) + '_stress_x'
                ] = pd.Series(stress_x_2nd)

            stress_dict[
                str(temp_params['thickness']/1000) + '_'
                + str(temp_params['time_1st']/60) + '_'
                + str(temp_params['time_2nd']/60) + '_stress_y'
                ] = pd.Series(stress_y_2nd)

        stress_df = pd.DataFrame(stress_dict)
        stress_df.to_csv(self.stress_args.output_file)
        return stress_df


if __name__ == '__main__':
    fem_solver = FemSolver()
    df = fem_solver.main()

