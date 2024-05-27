"""GNN logistic scheduler wrapper script. Can be used to convert task
description files into MARS schedules using pretrained GNN model.
Prints the generated plan into stdout."""

import json
import sys



from argparse import ArgumentParser

import yaml


argparser = ArgumentParser(description=__doc__)

argparser.add_argument('task_desc', default='./data/00374.yaml',
                       nargs='?',
                       help='default = "./tasks/00374.json".'
                       ' Task description file. JSON, YAML or'
                       ' legacy 4-file format')

argparser.add_argument('out', default='./schedules/schedule.json'
                       , nargs='?',
                       help='default = "./schedules/schedule.json".'
                       'Optional filepath to save schedule to.'
                       ' Will always be valid JSON/YAML. Type "-" to'
                       ' print to stdout.')

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
schedule = sch.schedule(args.task_desc)
if args.out.strip() == '-':
    json.dump(schedule, sys.stdout)
elif args.out.lower().endswith('.yaml'):
    with open(args.out, 'wt', encoding='utf-8') as f:
        yaml.safe_dump(schedule, f)
else:
    with open(args.out, 'wt', encoding='utf-8') as f:
        json.dump(schedule, f)
print('Done', file=sys.stderr)

