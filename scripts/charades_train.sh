
config=charades
gpus=0,1
gpun=2

master_addr=127.0.0.2
master_port=29504


config_file=configs/$config\.yaml
output_dir=outputs/$config

CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.launch \
--nproc_per_node=$gpun --master_addr $master_addr --master_port $master_port train_net.py --config-file $config_file OUTPUT_DIR $output_dir \
