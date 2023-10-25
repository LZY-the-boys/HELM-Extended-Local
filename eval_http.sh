source activate crfm-helm
set -e pipefail

: ${PORT:=8080}
: ${SUITE:=tmp}
: ${NAME:=test}
: ${EVALNUM:=50}
: ${OUTPUT:="/home/lzy/nips/submit/metrics"}

if [[ "$CONF" =~ .*summary.* ]]; then
    CONF=run_summary.conf
elif [[ "$CONF" =~ .*truthfulqabbq2.* ]]; then
    CONF=run_truthfulqabbq2.conf
elif [[ "$CONF" =~ .*mmlu2.* ]]; then
    CONF=run_mmlu2.conf
elif [[ "$CONF" =~ .*mmlu.* ]]; then
    CONF=run_mmlu.conf
elif [[ "$CONF" =~ .*bbq.* ]]; then
    CONF=run_bbq.conf
elif [[ "$CONF" =~ .*truthfulqa.* ]]; then
    CONF=run_truthfulqa.conf
elif [[ "$CONF" =~ .*unseen.* ]]; then
    CONF=run_unseen.conf
elif [[ "$CONF" =~ .*math.* ]]; then
    CONF=run_math.conf
elif [[ "$CONF" =~ .*1st.* ]]; then
    CONF=run_1st.conf
else
    CONF=run_http.conf
fi

echo ">>> use $CONF"
echo "OUTPUT $OUTPUT"

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

# write output to summary in the end
if [ "$SHOW" ];then
    python -m helm.benchmark.presentation.summarize --suite $SUITE
    python nips_metrics.py --suite $SUITE --output-path $OUTPUT
fi