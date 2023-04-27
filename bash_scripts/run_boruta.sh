#!/bin/sh

python3 boruta_ml.py --df "vectorised/dosenbach160_vectorised.csv.gz" --atlas "dosenbach160" --no-filtered
python3 boruta_ml.py --df "vectorised/ez_vectorised.csv.gz"  --atlas "ez" --no-filtered
python3 boruta_ml.py --df "vectorised/tt_vectorised.csv.gz"  --atlas "tt" --no-filtered
python3 boruta_ml.py --df "vectorised/cc200_vectorised.csv.gz"  --atlas "cc200" --no-filtered
python3 boruta_ml.py --df "vectorised/aal_vectorised.csv.gz"  --atlas "aal" --no-filtered
python3 boruta_ml.py --df "vectorised/cc400_vectorised.csv.gz"  --atlas "cc400" --no-filtered