#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters, 'split_name'"
    exit
fi

split_name=$1

trap exit SIGINT

hydrated="data/splits/${split_name}/hydrated"
stories="data/splits/${split_name}/stories"
for dir in "test" "train"; do
    tweets_dir="$hydrated/$dir"
    destination_dir="$stories/$dir"
    ratings=($(ls $tweets_dir | sort))
    for rating in ${ratings[@]}; do
        echo $rating
        mkdir -p $destination_dir/$rating
        folders=($(ls "$tweets_dir/$rating" | sort))
        for folder_name in ${folders[@]}; do
            echo $folder_name
            python preprocessing/build_stories.py -i "$tweets_dir/$rating/$folder_name/tweets" -o "$destination_dir/$rating/" --len 6
        done
    done
done
