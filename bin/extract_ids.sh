if [ "$#" -ne 4 ]; then
    echo "Illegal number of parameters, 'tweets_dir', 'output_name', 'count_per_spread', 'nb_spreads'"
    exit
fi

trap exit INT

tweets_dir=$1
output_name=$2
count_per_spread=$3
nb_spreads=$4

for rating in $(ls $tweets_dir)
do
    output_file="${output_name}_${rating}.csv"
    [ ! -f $output_file ] && echo "id,news_url,title,tweet_ids" > $output_file
    input_files=($(ls "$tweets_dir/$rating" | grep .csv))
    for file in ${input_files[@]::nb_spreads}
    do
        echo $file
        python fetch/tweets/extract_ids.py -i "${tweets_dir}/${rating}/${file}" -o $output_file -c $count_per_spread
    done
done

