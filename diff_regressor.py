import argparse
from c_profile_fitting import *
from numerical_method import *
import datetime
import json
from scipy.optimize import least_squares
from timeit import default_timer as timer
from stress_profile_gen import *
from logconf import logging
from timeit import default_timer as timer

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class DiffReg:
    def __init__(self):
        parser = argparse.ArgumentParser()

#        parser.add_argument('--file_path', default='./data/diff_reg/input.xlsx',
#                            help='file destination of edx profile')
#         parser.add_argument('--method', default='differential_evolution', const='differential_evolution',
#                             nargs='?', choices=['differential_evolution', 'shgo', 'basin_hopping', 'brutal', 'least_square'],
#                             help='choice of optimization method')

        parser.add_argument('--fitting_param_path', default='./data/diff_reg/input/fitting_param.txt')
        parser.add_argument('--repeated_times', type=int, default=10, help='define the times of regression')
        parser.add_argument('--output_file', default='./data/diff_reg/output/')
        parser.add_argument('--edx_data_mode', default='normal', const='normal', nargs='?',
                            choices=['normal', 'simple'], help='if edx data contains two steps')
        parser.add_argument('--comments', type=str, default='none', help='add comments to describe the IOX')

        self.reg_args = parser.parse_args('')
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.cost_func = self.init_cost_func()
        self.reg_params = self.init_reg_params()

    def init_cost_func(self):
        if self.reg_args.edx_data_mode == 'normal':
            cost_func = cost_func_2_steps
        else:
            cost_func = cost_func_2_steps_simple
        return cost_func

    def init_reg_params(self):
        with open(self.reg_args.fitting_param_path) as f:
            reg_params = json.load(f)
        return reg_params

    def do_regression(self):

        res = main_regression(self.cost_func, **self.reg_params)

        return res

    def main(self):

        log.info("Starting {}, {}".format(type(self).__name__, self.reg_args))

        for i in range(self.reg_args.repeated_times):

            log.info("starting {}/{} for Differential Evolution".format(
                i+1, self.reg_args.repeated_times
            ))

            start = timer()
            res = self.do_regression()
            output_file = os.path.join(
                self.reg_args.output_file, self.reg_args.comments + self.time_str + '.txt'
            )
            with open(output_file, 'a+') as f:
                f.write('\n' + 'Round{}:'.format(i) + '\n' + str(res) + '\n')
                f.close()
            end = timer()
            print('DE_optimization consumes {} minutes'.format((end - start) / 60))


def stress_cost_func(param, c_x, c_k_y, c_na_y, stress_x, stress_y):

    pass


if __name__ == '__main__':
    diff_reg = DiffReg()
    diff_reg.main()



