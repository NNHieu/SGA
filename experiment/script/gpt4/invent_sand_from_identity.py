from pathlib import Path
import subprocess
import os
import sys
from datetime import datetime

from sga.utils import get_root, get_script_parser, dict_to_cmds

root = get_root(__file__)


def main():
    python_path = Path(sys.executable).resolve()
    program_path = root / 'entry' / 'agent.py'
    base_cmds = [python_path, program_path]

    base_args = get_script_parser().parse_args()
    base_args = vars(base_args)

    base_args['overwrite'] = True

    my_env = os.environ.copy()
    # my_env['CUDA_VISIBLE_DEVICES'] = str(base_args['gpu'])

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    for seed in range(5):
        args = base_args | {
            'seed': seed,
            'path': f'gpt4/invent_sand_from_identity/{seed:04d}',
            'dataset_path': 'dataset/sand',
            'llm.primitives': '(identity)',
            'llm.entry': 'plasticity',
            'physics.env.physics': 'identity',
            'physics.env.physics.elasticity': 'sigma',
            'optim.alpha_position': 1e4,
            'optim.alpha_velocity': 1e1,
        }

        cmds = base_cmds + dict_to_cmds(args)
        str_cmds = [str(cmd) for cmd in cmds]
        subprocess.run(str_cmds, shell=False, check=False, env=my_env)

if __name__ == '__main__':
    main()
