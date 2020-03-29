if [ "$#" -lt 5 ]; then
    echo "Illegal number of parameters, give 'model_path', 'split_name', 'model_class', 'data_type' and 'output_name'"
    exit
fi

trap exit INT

model_path=$1
split_name=$2
model_class=$3
data_type=$4
output_name=$5

python_file="test.py"
if [ ${data_type} == "stories" ]; then
    python_file="test_bert.py"
fi;

original_split_name=$(echo ${split_name} | cut -c1-3)
python "modeling/${python_file}" --model-path "${model_path}" --model-class "${model_class}" -i "data/splits/${original_split_name}/${data_type}/test/" -o "data/splits/${split_name}/logs/${output_name}" -w 4  #  --gcn 1024 --fc 128
