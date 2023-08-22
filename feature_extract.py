import fink_science.slsn.kernel as k_slsn
import fink_science.slsn.feature_extraction as fe_slsn
import fink_science.slsn.classifier as clf_slsn

import fink_science.agn.kernel as k_agn
import fink_science.agn.feature_extraction as fe_agn
import fink_science.agn.feature_extraction as fe_agn
import fink_science.agn.classifier as clf_agn

import pandas as pd



def run_fex(data, target, file_name):

    source = 'ELASTICC'
    formated = fe_agn.format_data(data, source)
    
    print('Data formatted')

    if target == 'AGN':
        all_transformed, valid = fe_agn.transform_data(formated, k_agn.MINIMUM_POINTS, source)
        all_empty = [i.empty for i in all_transformed]
        all_features = fe_agn.parametrise(all_transformed, source)
        features = fe_agn.merge_features(all_features, k_agn.MINIMUM_POINTS, source)


    elif target == 'SLSN':
        all_transformed, valid = fe_slsn.transform_data(formated)
        all_empty = [i.empty for i in all_transformed]
        all_features = fe_slsn.parametrise(all_transformed)
        features = fe_slsn.merge_features(all_features)
        
    else:
        print("Only AGN or SLSN accepted !!")


    features.to_parquet(f'{file_name}_{target}_features.parquet')


if __name__ == "__main__":

    target = "SLSN" #SLSN
    dataset = "test" #train
    data = pd.read_parquet(f"{dataset}_sample.parquet")

    run_fex(data, target, dataset)
    