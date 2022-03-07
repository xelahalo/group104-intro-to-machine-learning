from turtle import pos
import pandas as pd

DATA_PATH = 'data\\breast-cancer-wisconsin\\wpbc.data'

headers = ['id', 'outcome', 'time']
features = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension']
# prefixes = ['mean', 'std_error', 'worst']
prefixes = ['M', 'SE', 'W']
headers_tail = ['tumor_size', 'lymph_node_status']

for prefix in prefixes:
    for feature in features:
        headers.append(prefix + '_' + feature)
# for postfix in range(1,4):
#         for feature in features:
#             headers.append(feature + str(postfix))
headers.extend(headers_tail)

data = pd.read_csv(DATA_PATH, names=headers, sep=',', na_values='?', index_col=False)

# Filling up the 4 missing lymph node status values with the mean of the column
data = data.fillna(data.mean(numeric_only=True))

print(data.head())
# N = 0
# R = 1
# data['outcome'].replace({'N': 0, 'R': 1}, inplace=True)