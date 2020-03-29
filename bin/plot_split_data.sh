#!/bin/bash

trap exit SIGINT

for split_name in $(ls "data/splits/" | grep -v test); do
    stats_dir="data/splits/${split_name}/stats"
    echo $stats_dir
    python visualization/plot_data_stats.py -i "${stats_dir}" --len 10
done
