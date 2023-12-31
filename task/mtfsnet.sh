/home/servicing/.conda/envs/maxwell/bin/python -u main.py \
--model "mtfsnet" \
--loss_fn "gradnorm" \
--batch_size 64 \
--cuda '0' \
--epochs 100 \
--step_size 10 \
--lr 1.e-4 \
--optim "Adam" \
--version "mtfsnet" \
--trainer "mtfsnet" \
--lr_scheduler 'warmup' \
--gamma 0.8