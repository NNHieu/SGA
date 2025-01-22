from pathlib import Path
import subprocess
import os
import sys
from datetime import datetime
import yaml

from sga.utils import get_root, get_script_parser, dict_to_cmds
from sga.sr.datamodules import get_datamodule

root = get_root(__file__)


def main():
    python_path = Path(sys.executable   ).resolve()
    program_path = root / 'entry' / 'agent_sr.py'
    base_cmds = [python_path, program_path]

    parser = get_script_parser()
    parser.add_argument('--ds_name', type=str)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=None)
    parser.add_argument('--llm_config_path', type=str)
    parser.add_argument('--llm_port', type=int, default=10000)
    base_args = parser.parse_args()

    start_idx = base_args.start_idx
    end_idx = base_args.end_idx
    llm_config_path = Path(base_args.llm_config_path)
    with open(llm_config_path) as f:
        llm_configs = yaml.safe_load(f)


    ds_name = base_args.ds_name
    if ds_name == "bio":
        data_path = 'datasets/bio-pop-growth-01142025_split'
    elif ds_name == "chem":
        data_path = 'datasets/chem-react-kinetics_01142025_split'
    elif ds_name == "mat":
        data_path = 'datasets/matsci-ss-01142025_split'
    elif ds_name == "phy":
        data_path = 'datasets/phys-oscillator-01142025_split'
    elif ds_name == "feynman":
        data_path = 'feynman'
    elif ds_name == "invfeynman":
        data_path = 'invfeynman'
    else:
        raise ValueError

    dm = get_datamodule(data_path)
    dm.setup()


    base_args = vars(base_args)
    base_args['overwrite'] = True
    del base_args['ds_name'], base_args['start_idx'], base_args['end_idx']


    my_env = os.environ.copy()
    my_env['CUDA_VISIBLE_DEVICES'] = str(base_args['gpu'])

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    ds_run_path = Path(f'experiment/log/{llm_configs["name"]}/{ds_name}/')

    # visited = [ f.name for f in ds_run_path.iterdir()  if f.is_dir()]
    visited = [f.parent.parent.parent.parent.name for f in list(ds_run_path.glob("*/0000/iteration/0005/all.json"))] # Finished runs
    print(visited)
    
    if end_idx is None:
        end_idx = len(dm.problems)
    for problem_index in range(start_idx, end_idx):
        problem = dm.problems[problem_index]
        if problem.equation_idx in visited:
            print(f"Skip problem {problem.equation_idx}: {problem.gt_equation.expression}")
            continue

        print(f"Searching exression for problem {problem.equation_idx}: {problem.gt_equation.expression}")

        seed = 0
        args = base_args | {
            'seed': seed,
            'path': (ds_run_path / f'{problem.equation_idx}' / f'{seed:04d}'),
            'dataset_path': data_path,
            'problem_index': f'{problem_index}',
            'llm.primitives': '(linear)',
            'llm.entry': 'sr',
            'physics.env.physics': 'linear',
        }

        cmds = base_cmds + dict_to_cmds(args)
        str_cmds = [str(cmd) for cmd in cmds]
        subprocess.run(str_cmds, shell=False, check=False, env=my_env)

if __name__ == '__main__':
    main()
