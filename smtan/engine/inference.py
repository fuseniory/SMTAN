import logging
import torch
from smtan.data.datasets.evaluation import evaluate
from smtan.utils.metric_logger import MetricLogger
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str


def compute_on_dataset(model, data_loader, device, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    meters = MetricLogger(delimiter="  ")
    for batch in data_loader:
        batches, idxs = batch
        with torch.no_grad():
            if timer:
                timer.tic()
            _,_,contrastive_output, iou_output, loss = model(batches.to(device))
            meters.update(loss_vid=loss[0].detach(), loss_sent=loss[1].detach(), loss_iou_stnc=loss[2].detach(), loss_iou_phrase=loss[3].detach(), scoremap_loss_pos=loss[4].detach(), scoremap_loss_neg=loss[5].detach())
            if timer:
                if not device.type == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            contrastive_output, iou_output = [o.to(cpu_device) for o in contrastive_output], [o.to(cpu_device) for o in iou_output]
        results_dict.update(
            {video_id: {'contrastive': result1, 'iou': result2} for video_id, result1, result2 in zip(idxs, contrastive_output, iou_output)}
        )
    logger = logging.getLogger("smtan.inference")
    logger.info(str(meters))
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    idxs = list(sorted(predictions.keys()))
    if len(idxs) != idxs[-1] + 1:
        logger = logging.getLogger("smtan.inference")
        logger.warning(
            "Number of samples that were gathered from multiple processes is not "
            "a contiguous set. Some samples might be missing from the evaluation"
        )

    predictions = [predictions[i] for i in idxs]
    return predictions

def inference(
        cfg,
        model,
        data_loader,
        dataset_name,
        nms_thresh,
        device="cuda",
    ):
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("smtan.inference")
    results = []
    for d_loader in data_loader:
        dataset = d_loader.dataset
        logger.info("Start evaluation on {} (Size: {}).".format(dataset.ann_name, len(dataset)))
        inference_timer = Timer()
        predictions = compute_on_dataset(model, d_loader, device, inference_timer)
        synchronize()
        total_infer_time = get_time_str(inference_timer.total_time)
        logger.info(
            "Model inference time: {} ({:.03f} s / inference per device, on {} devices)".format(
                total_infer_time,
                inference_timer.total_time * num_devices / len(dataset),
                num_devices,
            )
        )

        predictions = _accumulate_predictions_from_multiple_gpus(predictions)
        results.append(evaluate(cfg, dataset=dataset, predictions=predictions, nms_thresh=nms_thresh))
    return results
