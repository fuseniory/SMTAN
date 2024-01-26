

config_file=configs/activitynet.yaml
weight_dir=outputs/activitynet/
weight_file=outputs/activitynet/pool_model_7e.pth


batch_size=20

gpus=1

gpun=3

master_addr=127.0.0.2
master_port=28578

CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.launch \
--nproc_per_node=$gpun --master_addr $master_addr --master_port $master_port \
test_net.py --config-file $config_file --ckpt $weight_file OUTPUT_DIR $weight_dir TEST.BATCH_SIZE $batch_size

