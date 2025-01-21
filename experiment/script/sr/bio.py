from pathlib import Path
import subprocess
import os
import sys
from datetime import datetime

from sga.utils import get_root, get_script_parser, dict_to_cmds

root = get_root(__file__)


def main():
    python_path = Path(sys.executable).resolve()
    program_path = root / 'entry' / 'agent_sr.py'
    base_cmds = [python_path, program_path]

    parser = get_script_parser()
    parser.add_argument('--llm', type=str, default='openai-gpt-4-1106-preview', choices=[
        'openai-gpt-4-1106-preview',
        'openai-gpt-3.5-turbo-0125',
        'mistral-open-mixtral-8x7b',
        'anthropic-claude-3-sonnet-20240229',
    ])
    base_args = parser.parse_args()
    base_args = vars(base_args)

    base_args['overwrite'] = True

    my_env = os.environ.copy()
    my_env['CUDA_VISIBLE_DEVICES'] = str(base_args['gpu'])

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    for seed in range(5):
        args = base_args | {
            'seed': 0,
            'path': f'{base_args["llm"]}/sr_bio/{seed:04d}',
            'dataset_path': 'datasets/bio-pop-growth-01142025_split',
            'problem_index': f'{seed}',
            'llm.primitives': '(linear)',
            'llm.entry': 'sr',
            'physics.env.physics': 'linear',
        }

        cmds = base_cmds + dict_to_cmds(args)
        str_cmds = [str(cmd) for cmd in cmds]
        subprocess.run(str_cmds, shell=False, check=False, env=my_env)

if __name__ == '__main__':
    main()
