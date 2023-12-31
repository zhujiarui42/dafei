/home/servicing/projects/environments/anaconda3/envs/py37_torch_jiarui/bin/python -u main.py \
--model "carunet" \
--loss_fn "mse" \
--batch_size 16 \
--cuda '1' \
--epochs 100 \
--step_size 10 \
--lr 1.e-4 \
--optim "Adam" \
--version "carunet" \
--lr_scheduler 'warmup' \
--gamma 0.8