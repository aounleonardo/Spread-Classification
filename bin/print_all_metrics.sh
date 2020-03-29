if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters, give 'logs_path'"
    exit
fi

trap exit INT


logs_path=$1

cd ${logs_path}
checkpoints=($(ls | grep checkpoint | grep -v \.pth))

for checkpoint in ${checkpoints[@]}
do
    echo $checkpoint
    cat ${checkpoint}/*/metrics.json
    echo ""
    echo ""
done

