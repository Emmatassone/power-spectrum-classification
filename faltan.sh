#!/bin/bash

python3.10 PSC.py --RF --batch_size=64 --n_jobs=30 --n_estimators=200 --min_samples_leaf=50 --min_samples_split=25

python3.10 PSC.py --RF --batch_size=64 --n_jobs=30 --n_estimators=200 --min_samples_leaf=50 --min_samples_split=50

python3.10 PSC.py --RF --batch_size=64 --n_jobs=30 --n_estimators=200 --min_samples_leaf=50 --min_samples_split=100

