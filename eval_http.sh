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
# hack to 127.0.0.1:8080
python -m helm.benchmark.run \
    --conf-paths run_http.conf \
    --suite $SUITE \
    --max-eval-instances 50 \
    --num-threads 1 \
    --name $NAME \
    --url "http://127.0.0.1:$PORT"

python -m helm.benchmark.presentation.summarize --suite $SUITE
python nips_metrics.py --suite $SUITE