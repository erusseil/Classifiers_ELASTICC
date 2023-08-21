# Classifiers_ELASTICC
The classifiers are trained using ELASTICC data (v2) from 27 November 2023 to 31 January 2024. It corresponds to ~2.4 M alerts.
This data can be acquiered using the Fink Data Transfer Service : https://fink-portal.org/download

The reformat.sh and reformat.py files are used to compress the data by keeping only useful columns and concatenating them.

Create correct test samples
We use read_data.py for both AGN and SLSN to create a training and a testing sample. We select 500 K alerts in each sample. 

