#!/bin/bash

trap exit SIGINT

for split_name in $(ls "data/splits/" | grep -v test); do
    hydrated="data/splits/${split_name}/hydrated"
    stats="data/splits/${split_name}/stats"
    mkdir -p $stats
    for dir in "test" "train"; do
        stats_file="$stats/$dir.json"
        echo "$hydrated/$dir" - $stats_file
        python visualization/get_data_stats.py -i "$hydrated/$dir" -f "data/followers" -o "${stats_file}" -p 8 --len 10
    done
done
