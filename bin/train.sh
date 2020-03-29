if [ "$#" -lt 4 ]; then
    echo "Illegal number of parameters, give 'split_name', 'model_name', 'data_name', 'epochs' and 'with_nlp'"
    exit
fi

trap exit INT

split_name=$1
model_name=$2
data_name=$3
epochs=$4
with_nlp=$5

start_visdom() {
    echo "checking if visdom is running"
    (lsof -i | grep -q "visdom") && echo "visdom is running" && return
    echo "visdom is not running, will run ..."
    visdom &
    sleep 5
    echo "running visdom"
}

model_class=""
if [ $with_nlp == "y" ]; then
    model_class="graph"
elif [ $with_nlp == "n" ]; then
    model_class="graph_without_text"
else
    echo "'with_nlp' can either be 'y' or 'n', not $with_nlp"
    exit
fi

start_visdom

mkdir "data/splits/${split_name}/logs/$model_name"
python modeling/train.py -i "data/splits/${split_name}/${data_name}/train" -e $epochs -o "data/splits/${split_name}/logs/$model_name" --model-name $model_name --optimizer adam --lr 0.005 --gamma 0.9 --lr-update-every 1 --restart-every 10 --restart-factor 1.0 --init-lr-factor 1.0 --lr-reduce-patience 20 --lr-reduce-factor 0.5 --early-stop-patience 40 --weight-decay 0.0 -t 64 -v 32 --model-class "${model_class}" --gcn 1024 --fc 128
