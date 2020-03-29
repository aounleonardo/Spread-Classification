#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters, 'split_name', 'text_encoder', 'followers_dir'"
    exit
fi

split_name=$1
text_encoder=$2
followers_dir=$3

trap exit SIGINT

followers_dir_name=$(echo ${followers_dir} | grep -o "\bless_followers/.*\b" | cut -d "/" -f 2)
if [ -z "$followers_dir_name" ]; then
    exit
fi

hydrated="data/splits/${split_name}/hydrated"
cascades="data/splits/${split_name}/cascades_${followers_dir_name}"
for dir in "train"; do
    tweets_dir="$hydrated/$dir"
    destination_dir="$cascades/$dir"
    ratings=($(ls $tweets_dir | sort))
    for rating in ${ratings[@]}; do
        echo $rating
        mkdir -p $destination_dir/$rating
        folders=($(ls "$tweets_dir/$rating" | sort))
        for folder_name in ${folders[@]}; do
            echo $folder_name
            python preprocessing/build_cascades.py -i "$tweets_dir/$rating/$folder_name/tweets" -o "$destination_dir/$rating/" --len 6 -e "${text_encoder}" -f "${followers_dir}"
        done
    done
done
