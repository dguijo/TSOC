#!/usr/bin/env bash

source activate TSOC

datasets=("DistalPhalanxOutlineAgeGroup" "DistalPhalanxTW" "EthanolLevel" "MiddlePhalanxOutlineAgeGroup" "MiddlePhalanxTW" "ProximalPhalanxOutlineAgeGroup" "ProximalPhalanxTW")

dir=/home/dguijo/ArtTSOC/Outputs_Standard/

mkdir -p $dir;

for dataset in ${datasets[@]:0:7}
do
	for semilla in {0..9}
	do
		output="$dataset$semilla"
		nohup python -u cluster.py -t "/home/dguijo/ArtTSOC/timeseries/" -p "/home/dguijo/ArtTSOC/datasets" -r "/home/dguijo/ArtTSOC/results" -s "Standard" -d $dataset -n $semilla > "$dir$output" &
	done
done

dir=/home/dguijo/ArtTSOC/Outputs_RegLin/
mkdir -p $dir;

for dataset in ${datasets[@]:0:7}
do
	for semilla in {0..9}
	do
		output="$dataset"
		nohup python -u cluster.py -t "/home/dguijo/ArtTSOC/timeseries/" -p "/home/dguijo/ArtTSOC/datasets" -r "/home/dguijo/ArtTSOC/results" -s "RegLin" -d $dataset -n $semilla > "$dir$output" &
	done
done

conda deactivate