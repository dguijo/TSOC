#!/usr/bin/env bash

source activate TSOC

datasets=("DistalPhalanxOutlineAgeGroup" "DistalPhalanxTW" "EthanolLevel" "MiddlePhalanxOutlineAgeGroup" "MiddlePhalanxTW" "ProximalPhalanxOutlineAgeGroup" "ProximalPhalanxTW")

dir=/home/dguijo/ArtTSOC/outputs/
mkdir -p $dir;

for dataset in ${datasets[@]:0:7}
do
	output="output$dataset"
	nohup python -u cluster.py -t "/home/dguijo/ArtTSOC/timeseries/" -p "/home/dguijo/ArtTSOC/datasets/" -r "/home/dguijo/ArtTSOC/results/" -s "Ordinal_1" -d $dataset > "$dir$output" &
done

conda deactivate