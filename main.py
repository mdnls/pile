import argparse
import yaml
import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax.random as jr
from tqdm import tqdm
import seaborn as sns
import numpy as np
import shutil
import cheb
from run import run
import copy
def main(config_fname, overwrite=False):
    with open(os.path.join('configs', config_fname), "r") as f_in:
        config = yaml.safe_load(f_in)

    key = jr.key(config['key'])
    name = config['name']
    if not os.path.exists(f'expts/{name}'):
        os.makedirs(f'expts/{name}')
        with open(os.path.join(f'expts/{name}/config.yml'), "w+") as f_out:
            yaml.dump(config, f_out)
    elif overwrite:
        shutil.rmtree(f'expts/{name}')
        os.makedirs(f'expts/{name}')
        with open(os.path.join(f'expts/{name}/config.yml'), "w+") as f_out:
            yaml.dump(config, f_out)
    else:
        print(f'Experiment {name} exists and overwrite is false. Skipping...')
        return


    reg = config['train']['reg']
    PINN_reg_vals = jnp.linspace(*reg['PINN'])
    DATA_reg_vals = jnp.linspace(*reg['DATA'])

    i = 0
    for p in PINN_reg_vals:
        for d in DATA_reg_vals:
            print(f"\n\n Config {name} - pinn reg = {p}, data reg = {d}")
            _config = copy.deepcopy(config)
            _config['train']['reg']['PINN'] = p
            _config['train']['reg']['DATA'] = d
            _config['run-id'] = i
            i += 1
            run(config=_config, key=key)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='config file name', required=False)
    parser.add_argument('--run_all', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    cfgs_to_run = []
    if args.run_all:
        cfgs_to_run.extend([x for x in os.listdir('configs') if x.endswith(".yml")])
    elif args.config is not None:
        cfgs_to_run.append(args.config)

    for c in cfgs_to_run:
        print(f'\n\n====== {c.split(".yml")[0]} ======\n\n')
        main(c, overwrite=args.overwrite)