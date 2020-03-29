if [ "$#" -lt 4 ]; then
    echo "Illegal number of parameters, give 'split_name', 'model_name', 'epochs' and 'model_class'"
    exit
fi

trap exit INT

split_name=$1
model_name=$2
epochs=$3
model_class=$4

start_visdom() {
    echo "checking if visdom is running"
    (lsof -i | grep -q "visdom") && echo "visdom is running" && return
    echo "visdom is not running, will run ..."
    visdom &
    sleep 5
    echo "running visdom"
}

model_cls=""
if [ $model_class == "bert" ]; then
    model_cls="text"
elif [ $model_class == "bert_untrained" ]; then
    model_cls="text_untrained"
elif [ $model_class == "fasttext" ]; then
    model_cls="fasttext"
elif [ $model_class == "fasttext_untrained" ]; then
    model_cls="fasttext_untrained"
else
    echo "'model_class' can either be 'bert', 'bert_untrained', 'fasttext', or 'fasttext_untrained', not $model_class"
    exit
fi

start_visdom

mkdir "data/splits/${split_name}/logs/$model_name"
python modeling/train_bert.py -i "data/splits/${split_name}/stories/train" -e $epochs -o "data/splits/${split_name}/logs/$model_name" --model-name $model_name --optimizer adamw --lr 2e-5 --gamma 0.9 --lr-update-every 1 --restart-every 10 --restart-factor 1.0 --init-lr-factor 1.0 --lr-reduce-patience 20 --lr-reduce-factor 0.5 --early-stop-patience 40 --weight-decay 0.0 -t 64 -v 32 --model-class "${model_cls}"
