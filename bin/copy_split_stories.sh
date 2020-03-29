#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters, 'from_split', 'to_split'"
    exit
fi

from_split=$1
to_split=$2

to_hydrated="data/splits/${to_split}/hydrated"
from_hydrated_train="data/splits/${from_split}/hydrated/train"
from_hydrated_test="data/splits/${from_split}/hydrated/test"
from_stories_train="data/splits/${from_split}/stories/train"
from_stories_test="data/splits/${from_split}/stories/test"

trap exit SIGINT

stories="data/splits/${to_split}/stories"
for dir in "test" "train"; do
    tweets_dir="$to_hydrated/$dir"
    destination_dir="$stories/$dir"
    ratings=($(ls $tweets_dir | sort))
    for rating in ${ratings[@]}; do
        echo $rating
        mkdir -p $destination_dir/$rating
        folders=($(ls "$tweets_dir/$rating" | sort))
        for folder_name in ${folders[@]}; do
            source_path=""
            if ls "$from_stories_train/$rating" | grep -q "${folder_name}_"; then
                source_path="${from_stories_train}/$rating"
            elif ls "$from_stories_test/$rating" | grep -q "${folder_name}_"; then
                source_path="${from_stories_test}/$rating"
            fi   
            if [[ ${source_path} ]]; then
                echo $folder_name
                cp ${source_path}/${folder_name}_*.txt "${destination_dir}/$rating/"
            fi
        done
    done
done
