#/bin/bash

cd data
cd td 
# Download labels for words, phones and speakers
mkdir phone
cd phone
rsync -av ultrasuite-rsync.inf.ed.ac.uk::ultrasuite/labels-uxtd-uxssd-upx/uxtd/phone_labels/lab/* .
cd ../

mkdir word
cd word
rsync -av ultrasuite-rsync.inf.ed.ac.uk::ultrasuite/labels-uxtd-uxssd-upx/uxtd/word_labels/lab/* .
cd ../

mkdir speaker
cd speaker
rsync -av ultrasuite-rsync.inf.ed.ac.uk::ultrasuite/labels-uxtd-uxssd-upx/uxtd/speaker_labels/lab/* .
cd ../../

cd ssd

mkdir phone
cd phone
rsync -av ultrasuite-rsync.inf.ed.ac.uk::ultrasuite/labels-uxtd-uxssd-upx/uxssd/phone_labels/lab/* .
rsync -av ultrasuite-rsync.inf.ed.ac.uk::ultrasuite/labels-uxtd-uxssd-upx/upx/phone_labels/lab/* .
cd ../

mkdir word
cd word
rsync -av ultrasuite-rsync.inf.ed.ac.uk::ultrasuite/labels-uxtd-uxssd-upx/uxssd/word_labels/lab/* .
rsync -av ultrasuite-rsync.inf.ed.ac.uk::ultrasuite/labels-uxtd-uxssd-upx/upx/word_labels/lab/* .
cd ../

mkdir speaker
cd speaker
rsync -av ultrasuite-rsync.inf.ed.ac.uk::ultrasuite/labels-uxtd-uxssd-upx/uxssd/speaker_labels/lab/* .
rsync -av ultrasuite-rsync.inf.ed.ac.uk::ultrasuite/labels-uxtd-uxssd-upx/upx/speaker_labels/lab/* .
cd ../../