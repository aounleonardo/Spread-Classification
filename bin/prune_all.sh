#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters, 'tweets_dir', 'destination_folder', 'min_branch_size'"
    exit
fi

trap exit SIGINT

tweets_dir=$1
destination_folder=$2
min_branch_size=$3

ratings=($(ls $tweets_dir | sort))
for rating in ${ratings[@]}
do
    echo $rating
    mkdir $destination_folder/$rating
    folders=($(ls "$tweets_dir/$rating" | sort))
    for folder_name in ${folders[@]}
    do
        echo $folder_name
        mkdir $destination_folder/$rating/$folder_name
        mkdir $destination_folder/$rating/$folder_name/tweets
        python fetch/tweets/prune_retweet_tree.py -i "$tweets_dir/$rating/$folder_name/tweets" -o "$destination_folder/$rating/$folder_name/tweets" --len $min_branch_size
    done
done
