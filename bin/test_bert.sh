if [ "$#" -lt 3 ]; then
    echo "Illegal number of parameters, give 'model_path', 'split_name' and 'output_name'"
    exit
fi

trap exit INT

model_path=$1
split_name=$2
output_name=$3

python modeling/test_bert.py --model-path "${model_path}" -i "data/splits/${split_name}/stories/test/" -o "data/splits/${split_name}/logs/${output_name}" -w 4