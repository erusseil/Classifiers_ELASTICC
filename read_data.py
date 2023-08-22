import pandas as pd
import numpy as np

nom = 'SLSN'
nalerts = 500000

if nom == 'SLSN':
    code = [2246, 2242]
elif nom == 'AGN':
    code = [2332]

for sample in ['test', 'train']:
    
    features = pd.read_parquet(f'Analysis/test_sample/{sample}_{nom}_features.parquet')
    targets = pd.read_parquet(f'Analysis/test_sample/{sample}_sample.parquet')
    features['type'] = targets['target']
    features = features.dropna()
    features = features.sample(n=nalerts)
    features = features.rename(columns={'object_id':'id'})
    column_order = ['id', 'type'] + list(features.keys())[1:-1]
    features = features.loc[:, column_order]
    features = features.sort_values('id', ignore_index=True)
    features['id'] = [int(str(i) + str(idx)) for idx, i in enumerate(features['id'])]
    features['type'] = np.where(features['type'].isin(code), 1, 0)

    features.to_parquet(f'input/{sample}_{nom}.parquet')