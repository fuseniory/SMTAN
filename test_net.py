import argparse
import os
import torch
from smtan.config import cfg
from smtan.data import make_data_loader
from smtan.engine.inference import inference
from smtan.modeling import build_model
from smtan.utils.checkpoint import TrmCheckpointer
from smtan.utils.comm import synchronize, get_rank
from smtan.utils.logger import setup_logger

def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser(description="Mutual Matching Network")
    parser.add_argument(
        "--config-file",
        default="configs/pool_128x128_k5l8_tacos.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    save_dir = ""
    logger = setup_logger("smtan", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)
    model = build_model(cfg)
    model.to(cfg.MODEL.DEVICE)
    model.eval()
    output_dir = cfg.OUTPUT_DIR
    checkpointer = TrmCheckpointer(cfg, model, save_dir=output_dir)
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    logger.info("load from %s"%ckpt)
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)
    dataset_names = cfg.DATASETS.TEST
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    _ = inference(
        cfg,
        model,
        data_loaders_val,
        dataset_name=dataset_names,
        nms_thresh=cfg.TEST.NMS_THRESH,
        device=cfg.MODEL.DEVICE,
    )
    synchronize()

if __name__ == "__main__":
    main()
