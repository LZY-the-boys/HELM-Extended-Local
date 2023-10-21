source activate crfm-helm
set -e pipefail
if [ ! "$PORT" ];then
    PORT=8080
fi
if [ ! "$SUITE" ];then
    SUITE=tmp
fi
if [ ! "$NAME" ];then
    NAME=test
fi
if [ ! "$OUTPUT" ];then
    OUTPUT=.
fi
if [[ "$CONF" =~ .*summary.* ]]; then
    CONF=run_summary.conf
elif [[ "$CONF" =~ .*mmlu.* ]]; then
    CONF=run_mmlu.conf
elif [[ "$CONF" =~ .*bbq.* ]]; then
    CONF=run_bbq.conf
elif [[ "$CONF" =~ .*truthfulqa.* ]]; then
    CONF=run_truthfulqa.conf
else
    CONF=run_http.conf
fi

echo ">>> use $CONF"

function wait_port_available() {
    local port="$1"
    while true; do
        if nc -z localhost $port; then
            echo "$port start"
            break
        fi
        sleep 5
    done
    sleep 1
}

wait_port_available $PORT

# hack to 127.0.0.1:8080
python -m helm.benchmark.run \
    --conf-paths $CONF \
    --suite $SUITE \
    --max-eval-instances 50 \
    --num-threads 1 \
    --name $NAME \
    --url "http://127.0.0.1:$PORT"

python -m helm.benchmark.presentation.summarize --suite $SUITE
python nips_metrics.py --suite $SUITE --output-path $OUTPUT