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
python3 rename_files.py "./data/td/wav" "td"
python3 rename_files.py "./data/td/transcripts" "td"
#Remove folders
cd ./data/td/wav
sudo rm -rf core-uxtd/
cd ../transcripts
sudo rm -rf core-uxtd/
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

python3 rename_files.py "./data/ssd/wav" "ssd"
python3 rename_files.py "./data/ssd/transcripts" "ssd"

#Remove folders
cd ./data/ssd/wav
sudo rm -rf core-uxssd/
cd ../transcripts
sudo rm -rf core-uxssd/
cd ../../../

# download labels
sh download_labels.sh
# Remove instructor audio + short utterances
cd ../../
python3 -m speech.data.preprocess_audio "./speech/data/data/ssd/wav"
python3 -m speech.data.preprocess_audio "./speech/data/data/td/wav"