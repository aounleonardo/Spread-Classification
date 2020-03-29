#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters, 'tweets_dir', 'destination_folder'"
    exit
fi

trap exit SIGINT

tweets_dir=$1
destination_folder=$2

ratings=($(ls $tweets_dir | sort))
for rating in ${ratings[@]}
do
    echo $rating
    mkdir $destination_folder/$rating
    folders=($(ls "$tweets_dir/$rating" | sort))
    for folder_name in ${folders[@]}
    do
        echo $folder_name
        python preprocessing/build_graph.py -i "$tweets_dir/$rating/$folder_name/tweets" -o "$destination_folder/$rating/$folder_name.bin" -f data/followers -c 200
    done
done
