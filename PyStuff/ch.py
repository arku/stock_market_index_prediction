import csv

with open('new_features.csv') as f:
    reader = csv.reader(f, delimiter=' ', skipinitialspace=True)
    first_row = next(reader)
    num_cols = len(first_row)
print num_cols, first_row
