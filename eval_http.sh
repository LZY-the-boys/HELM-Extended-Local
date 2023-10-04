source activate crfm-helm
set -e pipefail
if [ ! "$PORT" ];then
    PORT=8080
fi
if [ ! "$SUITE" ];then
    SUITE=tmp
fi
# hack to 127.0.0.1:8080
python -m helm.benchmark.run \
    --conf-paths run_http.conf \
    --suite $SUITE \
    --max-eval-instances 1 \
    --num-threads 1 \
    --url "http://127.0.0.1:$PORT"

python -m helm.benchmark.presentation.summarize --suite $SUITE
# Start a web server to display benchmark results
python -m helm.benchmark.server -p $PORT