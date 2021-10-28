CUDA_VISIBLE_DEVICES=2,3 python3 \
train.py --pretrained_root './pretrain'  --network 'sphnet' --output_dir './ckpt/test' --loss 'CosFace'  \
--batch_size 64 --num_client 40 --client_sampled_ratio 1.0 --lr 0.001 --total_round 12 --local_epoch 1 --fedface --init_fc \
--add_pretrained_data --combine_dataset  --contrastive_bb --return_all --BCE_local #--adaptive_local_epoch