#!/bin/bash
echo "Downloading Ultrasuite Data"
# Download wav files
mkdir data 
cd data
mkdir td
cd td
mkdir wav
cd wav
rsync -av --relative ultrasuite-rsync.inf.ed.ac.uk::ultrasuite/core-uxtd/core/*/*.wav .
cd ../
mkdir transcripts
cd transcripts
rsync -av --relative ultrasuite-rsync.inf.ed.ac.uk::ultrasuite/core-uxtd/core/*/*.txt .


cd ../../../
python rename_files.py "./data/td/wav" "td"
python rename_files.py "./data/td/transcripts" "td"
#Remove folders
cd ./data/td/wav
rm -rf core-uxtd/
cd ../transcripts
rm -rf core-uxtd/
cd ../../../
cd data 
mkdir ssd
cd ssd
mkdir wav
cd wav 
rsync -av --relative ultrasuite-rsync.inf.ed.ac.uk::ultrasuite/core-uxssd/core/*/*/*.wav .
rsync -av --relative ultrasuite-rsync.inf.ed.ac.uk::ultrasuite/core-upx/core/*/*/*.wav .
cd ../
mkdir transcripts
cd transcripts
rsync -av --relative ultrasuite-rsync.inf.ed.ac.uk::ultrasuite/core-uxssd/core/*/*/*.txt .
rsync -av --relative ultrasuite-rsync.inf.ed.ac.uk::ultrasuite/core-upx/core/*/*/*.txt .
cd ../../../

python rename_files.py "./data/ssd/wav" "ssd"
python rename_files.py "./data/ssd/transcripts" "ssd"

#Remove folders
cd ./data/ssd/wav
rm -rf core-uxssd/
cd ../transcripts
rm -rf core-uxssd/
cd ../../../