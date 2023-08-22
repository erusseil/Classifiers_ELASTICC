# Classifiers_ELASTICC
The classifiers are trained using ELASTICC data (v2) from 27 November 2023 to 31 January 2024. It corresponds to ~2.4 M alerts.
This data can be acquiered using the Fink Data Transfer Service : https://fink-portal.org/download

* __reformat.sh__ and __reformat.py__ files are used to compress the data by keeping only useful columns and concatenating the history.

* __sample.py__ is used to create a testing and a training sample (50/50) which ensures that objectId does not overlap between samples. 

* __read_data.py__ is used for both AGN and SLSN to create a training and a testing sample. We select 500 K alerts in each sample. 

* __feature_extract.py__ is used to extract both AGN and SLSN features from the training/testing samples.

* __read_data.py__ is used on training samples to create a file format ready to pass to the active learning step.

* Configure __AL_loop.py__ to run for 5000 loops querying 10 alerts at each step. The initial training size is 100 and we use 25 trees for the Random Forest during the AL phase (ensures fast process).
Use __start_AL.py__ and __start_AL__.sh to run 5 randomly initialised computations on 5 cores. Run for UncSampling and RandomSampling for both AGN and SLSN.

* The __metrics.ipynb__ notebook is used to visualise AL results and compute a final random forest model using the best of the 5 UncSampling run.
We use RandomForestClassifier(random_state=0, max_depth=15, min_samples_leaf=0.0001) to ensure an efficient yet light model.

* Finally __AGN_perf.ipynb__ and __SLSN_perf.ipynb__ are used to analyse the performances of the classifiers on the testing samples and produce the plots.

:)
