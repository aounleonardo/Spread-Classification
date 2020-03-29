if [ "$#" -lt 2 ]; then
	    echo "Illegal number of parameters"
	    exit
fi

trap exit SIGINT

origin_folder=$1
destination_folder=$2
begin=$3
count=$4

for rating in True False
do
    mkdir $destination_folder/$rating
    files=($(ls $origin_folder/$rating | sort | grep -oP "[^\.]*" | grep -v csv))
    for filename in ${files[@]:begin:count}
    do
        echo $filename
         python build_graph.py -i $origin_folder/$rating/$filename.csv -o $destination_folder/$rating/$filename.bin -d ../data -c 100
    done
done
