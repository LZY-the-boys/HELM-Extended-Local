tmux new -d -s eval3 'CUDA_VISIBLE_DEVICES=2 bash eval_nips3.sh'
tmux new -d -s eval4 'CUDA_VISIBLE_DEVICES=3 bash eval_nips4.sh'