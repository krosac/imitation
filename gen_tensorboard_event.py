import argparse
import csv
from collections import defaultdict
import os
import torch.utils.tensorboard as thboard

# Command line arguments
args = argparse.ArgumentParser(description='post-process training stats of inverse rl')
args.add_argument('--stats',metavar='STATS', type=str, nargs='?', default='data/pendulum/progress.csv', help='input stats')
args.add_argument('--fields',metavar='FIELDS', type=str, nargs='+', default=['all'], help='metrics to plot')
args.add_argument('--log_dir',metavar='OUTPUT_DIR', type=str, nargs='?', default='tensorboard/', help='directory to output tensorboard event')
args.add_argument('--print_fields_only',dest='print_fields_only', action='store_true')
args = args.parse_args()

filename = args.stats

# initializing the titles and rows list
fields = []
rows = []
  
# reading csv file
with open(filename, 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)
      
    # extracting field names through first row
    fields = next(csvreader)
  
    # extracting each data row one by one
    for row in csvreader:
        rows.append(row)
  
    # get total number of rows
    print("Total no. of rows: %d"%(csvreader.line_num))
    
# printing the field names
print('Field names are:' + ', '.join(field for field in fields))
if args.print_fields_only:
    exit(1)
    
if len(args.fields) == 1 and args.fields[0] == 'all':
    args.fields = fields
    
summary_dir = os.path.join(args.log_dir, "summary")
os.makedirs(summary_dir, exist_ok=True)
summary_writer = thboard.SummaryWriter(summary_dir)
iter_id = fields.index('Iteration')
dl = defaultdict(list)
for field in args.fields:
    id = fields.index(field)
    for r in range(1, len(rows)):
        value = float(rows[r][id])
        dl[field].append(value)
        summary_writer.add_scalar(field, value, int(rows[r][iter_id]))
    