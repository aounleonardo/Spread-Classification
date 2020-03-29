if [ "$#" -lt 2 ]; then
    echo "Illegal number of parameters, give 'output' and 'epochs'"
    exit
fi

trap exit INT

output=$1
epochs=$2

start_visdom() {
    echo "checking if visdom is running"
    (lsof -i | grep -q "visdom") && echo "visdom is running" && return
    echo "visdom is not running, will run ..."
    visdom &
    sleep 5
    echo "running visdom"
}

start_visdom

mkdir $output
python modeling/train.py -i "data/graphs/politifact_len200" -e $epochs -o $output --optimizer adam --lr 0.002 --gamma 1 --lr-update-every -1 --restart-factor 1.0 --init-lr-factor 1.0 --lr-reduce-patience -1 --lr-reduce-factor 0.5 --early-stop-patience -1 --weight-decay 0.015 --sizes 600
