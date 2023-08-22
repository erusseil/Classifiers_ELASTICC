import pandas as pd
import numpy as np
import random

targets = [2222, 2224, 2226, 2233, 2235, 2243, 2245, 2322, 2324, 2332, 2223, 2225, 2232, 2234, 2242, 2244, 2246, 2323, 2325]

subsample = []

for target in targets:
    agg = pd.read_parquet(f'/media/ELAsTICC/Fink/first_year/ftransfer_elasticc_v2_2023-08-12_245226/aggregated/classId={target}.parquet')
    agg['target'] = target
    subsample.append(agg)
    
subsample = pd.concat(subsample).sample(frac=1, ignore_index=True, random_state=0)

# Make sure that ObjectId don't overlap between samples
ids = np.unique(subsample['objectId'])
np.random.shuffle(ids)

subsample[subsample['objectId'].isin(ids[:int(len(ids)/2)])].reset_index(drop=True).to_parquet('test_sample.parquet')
subsample[subsample['objectId'].isin(ids[int(len(ids)/2):])].reset_index(drop=True).to_parquet('train_sample.parquet')