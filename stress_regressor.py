import argparse
import os
import pandas as pd

import matplotlib.pyplot as plt

import datetime
import json
from stress_profile_gen import dilation_coeff_reg
from logconf import logging


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class StressReg:

    def __init__(self,):
        parser = argparse.ArgumentParser()

        parser.add_argument('--fitting_param_path', default='./data/stress_reg/input/stress_fitting_param.txt')
        parser.add_argument('--input_data', default='./data/stress_reg/input/input.xlsx')
        parser.add_argument('--output_path', default='./data/stress_reg/output/')
        parser.add_argument(
            '--IOX_step', default='2nd', const='2nd', nargs='?', choices=['1st', '2nd'],
            help='describe the sample: 1st step IOXed OR 2nd step IOXed?'
        )
        parser.add_argument('--comments', type=str, default='none', help='add comments to describe the IOX')

        self.reg_args = parser.parse_args('')
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.data = self.init_input_data()
        self.stress_x = self.data.stress_x.dropna().values[:-2]
        self.stress_y = self.data.stress_y.dropna().values[:-2]
        self.reg_params = self.init_reg_param()

    def init_reg_param(self):
        with open(self.reg_args.fitting_param_path) as f:
            reg_params = json.load(f)
        return reg_params

    def init_input_data(self):
        return pd.read_excel(self.reg_args.input_data)

    def main(self):
        opt_x_ini = [1e-2, 1e-2]
        res, x_cal, stress_cal = dilation_coeff_reg(
            opt_x_ini, self.reg_params, (self.stress_x, self.stress_y), IOX_step=self.reg_args.IOX_step
        )
        fig, ax = plt.subplots(1,1)
        ax.scatter(self.stress_x, self.stress_y, s=10, alpha=0.2, facecolors='none', edgecolors='b',
                   label='measured stress')
        ax.plot(x_cal, stress_cal, lw=1, alpha=0.4, c='r', label='fitted stress')
        ax.grid(True)
        ax.set_xlabel('Depth (um)')
        ax.set_ylabel('Compressive stress(MPa)')
        ax.set_title('B_K = {:.6f}, \n B_Na = {:.6f}'.format(res.x[0], res.x[1]))
        ax.legend()
        fig.savefig(os.path.join(self.reg_args.output_path, 'output.png'))

        return res, x_cal, stress_cal, self.stress_x, self.stress_y


if __name__ == '__main__':
    stress_reg = StressReg()
    # stress_reg.reg_args.IOX_step = '1st'
    res, x_cal, stress_cal, x_true, stress_true = stress_reg.main()
    # plt.scatter(x_true, stress_true, s=10, alpha=0.2, facecolors='none', edgecolors='b',
    #             label='measured stress')
    # plt.plot(x_cal, stress_cal, lw=1, alpha=0.4, c='r', label='fitted stress')
    # plt.grid(True)
    # plt.xlabel('Depth (um)')
    # plt.ylabel('Compressive stress(MPa)')
    # plt.legend()

