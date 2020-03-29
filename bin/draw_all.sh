#!/bin/bash

if [ "$#" -ne 2 ]; then
	    echo "Illegal number of parameters, 'graphs_dir', 'images_dir'"
	    exit
fi

trap exit SIGINT

graphs_dir=$1
images_dir=$2

ratings=($(ls $graphs_dir | sort))
for rating in ${ratings[@]}
do
    echo $rating
    mkdir $images_dir/$rating
    files=($(ls "$graphs_dir/$rating" | sort | grep -oP "[^\.]*" | grep -v bin))
    for filename in ${files[@]}
    do
        echo $filename
         python visualization/draw_graph.py -i "$graphs_dir/$rating/$filename.bin" -o "$images_dir/$rating/$filename.png"
    done
done
