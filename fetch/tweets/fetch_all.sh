if [ "$#" -lt 1 ]; then
	    echo "Illegal number of parameters"
	    exit
fi

trap exit SIGINT

origin_folder=$1
begin=$2
count=$3

for rating in False True
do
    files=($(ls $origin_folder/$rating | sort | grep -oP "[^\.]*" | grep -v csv))
    for filename in ${files[@]:begin:count}
    do
        echo $filename
        python fetch_followers.py -i $origin_folder/$rating/$filename.csv -d ../../data --retries 4 --threads 16 -c 200
    done
done
