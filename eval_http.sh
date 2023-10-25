source activate crfm-helm
set -e pipefail
if [ ! "$PORT" ];then
    PORT=8080
fi
if [ ! "$SUITE" ];then
    SUITE=exl23bit
fi
if [ ! "$NAME" ];then
    NAME=test
fi
if [[ "$CONF" =~ .*summary.* ]]; then
    CONF=run_summary.conf
elif [[ "$CONF" =~ .*mmlu2.* ]]; then
    CONF=run_mmlu2.conf
elif [[ "$CONF" =~ .*mmlu.* ]]; then
    CONF=run_mmlu.conf
elif [[ "$CONF" =~ .*bbq.* ]]; then
    CONF=run_bbq.conf
elif [[ "$CONF" =~ .*truthfulqabbq2.* ]]; then
    CONF=run_truthfulqabbq2.conf
elif [[ "$CONF" =~ .*truthfulqa.* ]]; then
    CONF=run_truthfulqa.conf
elif [[ "$CONF" =~ .*1st.* ]]; then
    CONF=run_1st.conf
elif [[ "$CONF" =~ .*unseen.* ]]; then
    CONF=run_unseen.conf
elif [[ "$CONF" =~ .*math.* ]]; then
    CONF=run_math.conf
else
    CONF=run_nips.conf
fi
if [ ! "$OUTPUT" ];then
    OUTPUT=/home/ubuntu/lzy/submit/4090/metrics
fi

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
    --max-eval-instances $EVALNUM \
    --num-threads 1 \
    --name $NAME \
    --url "http://127.0.0.1:$PORT"

python -m helm.benchmark.presentation.summarize --suite $SUITE
python nips_metrics.py --suite $SUITE --output-path $OUTPUT
