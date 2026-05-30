
# python classifier.py --use_gpu --epochs 1 --fine-tune-mode last-linear-layer --lr 1e-3 --batch_size 64

python classifier.py \
    --use_gpu \
    --fine-tune-mode full-model \
    --lr 1e-5 \
    --epochs 10 \
    --batch_size 8 \
    --hidden_dropout_prob 0.2 \
    --sst-filepath sst-classifier.pt \
    --cfimdb-filepath cfimdb-classifier.pt \
    --predictions-prefix 01 \
> log.txt


python ./scripts/run_classifier.py \
    --use_gpu \
    --fine-tune-mode full-model \
    --lr 1e-5 \
    --epochs 10 \
    --batch_size 8 \
    --hidden_dropout_prob 0.2 \
    --sst-filepath sst-classifier.pt \
    --cfimdb-filepath cfimdb-classifier.pt \
    --predictions-prefix 01 \
> log.txt