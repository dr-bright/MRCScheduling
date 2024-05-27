"""GNN logistic scheduler wrapper script. Can be used to convert task
description files into MARS schedules using pretrained GNN model.
Prints the generated plan into stdout."""

import json
import sys



from argparse import ArgumentParser


argparser = ArgumentParser(description=__doc__)

argparser.add_argument('task_desc', default='./data/00374.yaml',
                       nargs='?',
                       help='default = "./data/00374.yaml".'
                       ' Task description file. JSON, YAML or'
                       ' legacy 4-file format')

argparser.add_argument('plan_out', default='-', nargs='?',
                       help='Optional filepath to save plan to.'
                       ' Will always be valid JSON/YAML')

argparser.add_argument('checkpoint_tar', default='./checkpoint.tar',
                       nargs='?',
                       help='default = "./checkpoint.tar".'
                       ' GNN model checkpoint')

args = argparser.parse_args()

print('Initialization...', file=sys.stderr)
from .utils import Scheduler

print('Loading GNN model...', file=sys.stderr)
sch = Scheduler(args.checkpoint_tar)
print('Running the scheduler...', file=sys.stderr)
plan = sch.schedule(args.task_desc)
if args.plan_out.strip() == '-':
    json.dump(plan, sys.stdout)
else:
    with open(args.plan_out, 'wt', encoding='utf-8') as f:
        json.dump(plan, f)
print('Done', file=sys.stderr)

