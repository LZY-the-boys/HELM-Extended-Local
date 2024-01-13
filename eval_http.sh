eval "$(conda shell.bash hook)"
conda activate crfm-helm
set -e pipefail

cd $home/HELM-Extended-Local

: ${PORT:=8080}
: ${SUITE:=tmp}
: ${NAME:=moe}
: ${EVALNUM:=50}
: ${OUTPUT:="$home/nips_submit/metrics"}

if [[ "$CONF" =~ .*summary.* ]]; then
    CONF=run_summary.conf
elif [[ "$CONF" =~ .*cnn.* ]]; then
    CONF=run_1st_cnn.conf
elif [[ "$CONF" =~ .*bigbench.* ]]; then
    CONF=run_bigbench.conf
elif [[ "$CONF" =~ .*truthfulqabbq2.* ]]; then
    CONF=run_truthfulqabbq2.conf
elif [[ "$CONF" =~ .*mmlu3.* ]]; then
    CONF=run_mmlu3.conf
elif [[ "$CONF" =~ .*all.* ]]; then    
    CONF=run_all.conf
elif [[ "$CONF" =~ .*mmlu2.* ]]; then
    CONF=run_mmlu2.conf
elif [[ "$CONF" =~ .*mmlu.* ]]; then
    CONF=run_mmlu.conf
elif [[ "$CONF" =~ .*truthfulqabbq2.* ]]; then
    CONF=run_truthfulqabbq2.conf
elif [[ "$CONF" =~ .*bigbench.* ]]; then
    CONF=run_bigbench.conf
elif [[ "$CONF" =~ .*bbq.* ]]; then
    CONF=run_bbq.conf
elif [[ "$CONF" =~ .*truthfulqa.* ]]; then
    CONF=run_truthfulqa.conf
elif [[ "$CONF" =~ .*1st.* ]]; then
    CONF=run_1st.conf
elif [[ "$CONF" =~ .*unseen.* ]]; then
    CONF=run_unseen.conf
elif [[ "$CONF" =~ .*math.* ]]; then
    CONF=run_math.conf
elif [[ "$CONF" =~ .*1st.* ]]; then
    CONF=run_1st.conf
elif [[ "$CONF" =~ .*moe.* ]]; then
    CONF=run_moe.conf
else
    CONF=run_http.conf
fi

echo ">>> use $CONF"
echo "OUTPUT $OUTPUT"

wait_port_available $PORT

# hack to 127.0.0.1:8080
T=$(date +%s)

python -m helm.benchmark.run \
    --conf-paths $CONF \
    --suite $SUITE \
    --max-eval-instances $EVALNUM  \
    --num-threads 1 \
    --name $NAME \
    --url "http://127.0.0.1:$PORT"

T="$(($(date +%s)-T))"
formatted_time=$(printf "time %02dd:%02dh:%02dmin:%02ds\n" "$((T/86400))" "$((T/3600%24))" "$((T/60%60))" "$((T%60))")
echo "[$(date '+%Y-%m-%d %H:%M:%S')][tmux-$CUDA_VISIBLE_DEVICES] $1 : [$formatted_time]"  >> $LZY_HOME/time.txt
# write output to summary in the end
if [ "$SHOW" ];then
    python -m helm.benchmark.presentation.summarize --suite $SUITE
    python nips_metrics.py --suite $SUITE --output-path $OUTPUT
fi
