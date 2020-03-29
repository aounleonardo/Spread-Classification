#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters, 'tweets_dir'"
    exit
fi

trap exit INT

tweets_dir=$1

ratings=($(ls $tweets_dir | sort))
for rating in ${ratings[@]}
do
    echo $rating
    folder_names=($(ls "$tweets_dir/$rating" | sort))
    folders=($(for folder in ${folder_names[@]}; do echo "$tweets_dir/$rating/$folder/tweets"; done))
    python fetch/tweets/fetch_followers.py -i ${folders[@]} -o data/followers -c 200 --retries 4 --threads 32
done
