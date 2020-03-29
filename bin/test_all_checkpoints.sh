if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters, give 'logs_path', 'model_class' and 'data_type'"
    exit
fi

trap exit INT


logs_path=$1
model_class=$2
data_type=$3

split_name=$(echo ${logs_path} | grep -o "\bsplits/.*/logs\b" | cut -d '/' -f 2)
run_name=$(echo ${logs_path} | grep -o "\b${split_name}/logs/.*" | cut -d '/' -f 3,4)

checkpoints=($(ls ${logs_path} | grep -o "checkpoint_[^\.]*"))

for checkpoint in ${checkpoints[@]}
do  
    echo $checkpoint
    model_path="${logs_path}/${checkpoint}.pth"
    output_name="${logs_path}/$checkpoint"
    echo $model_path
    bash bin/test.sh ${model_path} ${split_name} ${model_class} ${data_type} "${run_name}/$checkpoint"
done

