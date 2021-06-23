#!/bin/bash

echo "Downloading Ultrasuite Data"

# Download wav files
mkdir td
cd td
mkdir wav
rsync -av ultrasuite-rsync.inf.ed.ac.uk::ultrasuite/core-uxtd/core/*/*.wav .
cd ../
mkdir transcripts
cd transcripts
rsync -av ultrasuite-rsync.inf.ed.ac.uk::ultrasuite/core-uxtd/core/*/*.txt .

cd ../../
mkdir ssd
cd ssd
mkdir wav
cd wav 
rsync -av ultrasuite-rsync.inf.ed.ac.uk::ultrasuite/core-uxssd/core/*/*.wav .
rsync -av ultrasuite-rsync.inf.ed.ac.uk::ultrasuite/core-upxd/core/*/*.wav .
cd ../
mkdir transcripts
cd transcripts
rsync -av ultrasuite-rsync.inf.ed.ac.uk::ultrasuite/core-uxssd/core/*/*.txt .
rsync -av ultrasuite-rsync.inf.ed.ac.uk::ultrasuite/core-upxd/core/*/*.txt .
cd ../../


