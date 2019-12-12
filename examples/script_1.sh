#!/usr/bin/env bash

source activate TSOC

datasets=("Beef" "BME" "ChlorineConcentration" "CinCECGTorso" "DistalPhalanxOutlineAgeGroup" "DistalPhalanxTW" "ECG5000" "EthanolLevel" "Lightning7" "MiddlePhalanxOutlineAgeGroup" "MiddlePhalanxTW" "ProximalPhalanxOutlineAgeGroup" "ProximalPhalanxTW" "SmoothSubspace" "StarLightCurves" "UMD")

dir=/home/dguijo/ArtTSOC/outputs/
mkdir -p $dir;

for dataset in ${datasets[@]:0:16}
do
	output="output$dataset"
	nohup python main.py -t "/home/dguijo/ArtTSOC/timeseries/" -p "/home/dguijo/ArtTSOC/datasets/" -r "/home/dguijo/ArtTSOC/results/" -d $dataset > "$dir$output" &
done

conda deactivate