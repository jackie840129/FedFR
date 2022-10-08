CUDA_VISIBLE_DEVICES=0,1 python3 \
train.py --pretrained_root './pretrain'  --network 'sphnet' --output_dir './ckpt/FedFR' --loss 'CosFace'  \
--batch_size 64 --num_client 40 --client_sampled_ratio 1.0 --lr 0.001 --total_round 20 --local_epoch 10 --fedface \
--add_pretrained_data --combine_dataset  --contrastive_bb --return_all --BCE_local --adaptive_local_epoch