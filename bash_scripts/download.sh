#!/bin/sh


PARENT="$HOME/Documents/MscProject"
DF="$PARENT/phenotype_files/pheno_clean.csv"


python3 download_process.py --atlas "rois_dosenbach160" --df "$DF" --store "fc"
python3 download_process.py --atlas "rois_ez" --df "$DF" --store "fc"
python3 download_process.py --atlas "rois_ho" --df "$DF" --store "fc"
python3 download_process.py --atlas "rois_tt" --df "$DF" --store "fc"
python3 download_process.py --atlas "rois_cc400" --df "$DF" --store "fc"
python3 download_process.py --atlas "rois_cc200" --df "$DF" --store "fc"