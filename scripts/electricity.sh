export CUDA_VISIBLE_DEVICES=0,1

python -u main.py \
    --dataset electricity \
    --rlen 384 \
    --blen 96 \
    --qlen 96 \
    --moving_avg 169 \
    --lr 2e-3 \
    --batch_size 64

python -u main.py \
    --dataset electricity \
    --rlen 384 \
    --blen 96 \
    --qlen 192 \
    --moving_avg 169 \
    --lr 2e-3 \
    --batch_size 64

python -u main.py \
    --dataset electricity \
    --rlen 384 \
    --blen 96 \
    --qlen 336 \
    --moving_avg 169 \
    --lr 2e-3 \
    --batch_size 64

python -u main.py \
    --dataset electricity \
    --rlen 384 \
    --blen 96 \
    --qlen 720 \
    --moving_avg 169 \
    --lr 2e-3 \
    --batch_size 64