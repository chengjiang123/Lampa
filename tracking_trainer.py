#import nni
import yaml
import argparse
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
from torchmetrics import MeanMetric
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, StepLR
from utils import set_seed, get_optimizer, log, get_lr_scheduler, add_random_edge, get_loss
from torch_geometric.utils import unbatch
from utils.get_data import get_data_loader, get_dataset
from utils.get_model import get_model
from utils.metrics import acc_and_pr_at_k, point_filter
import math

import logging

def setup_logger(log_dir):
    logger = logging.getLogger('epoch_history')
    logger.setLevel(logging.INFO)
    
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_dir / 'epoch_history.txt')
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)
    
    # Create formatters and add them to handlers
    c_format = logging.Formatter('%(message)s')
    f_format = logging.Formatter('%(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)
    
    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    return logger

def train_one_batch(model, optimizer, criterion, data, lr_s):
    model.train()
    embeddings = model(data)
    #loss = criterion(embeddings, data.point_pairs_index, data.particle_id, data.reconstructable, data.pt)
    loss = criterion(embeddings, data.edge_index, data.particle_id, data.reconstructable, data.pt)
    
    if torch.isnan(loss):
        print("Loss is NaN, skipping backward")
        return float("nan"), embeddings.detach(), data.particle_id.detach()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if lr_s is not None and isinstance(lr_s, LambdaLR):
        lr_s.step()
    return loss.item(), embeddings.detach(), data.particle_id.detach()


@torch.no_grad()
def eval_one_batch(model, optimizer, criterion, data, lr_s):
    model.eval()
    embeddings = model(data)
    #loss = criterion(embeddings, data.point_pairs_index, data.particle_id, data.reconstructable, data.pt)
    loss = criterion(embeddings, data.edge_index, data.particle_id, data.reconstructable, data.pt)
    if torch.isnan(loss):
        print("Loss is NaN, skipping backward")
        return float("nan"), embeddings.detach(), data.particle_id.detach()
    return loss.item(), embeddings.detach(), data.particle_id.detach()


def process_data(data, phase, device, epoch, p=0.2):
    data = data.to(device)
    #if phase == "train":
        # pairs_to_add = add_random_edge(data.point_pairs_index, p=p, batch=data.batch, force_undirected=True)
    #    num_aug_pairs = int(data.point_pairs_index.size(1) * p / 2)
    #    pairs_to_add = to_undirected(torch.randint(0, data.num_nodes, (2, num_aug_pairs), device=device))
    #    data.point_pairs_index = torch.cat([data.point_pairs_index, pairs_to_add], dim=1)
    return data


def run_one_epoch(
    model,
    optimizer,
    criterion,
    data_loader,
    phase,
    epoch,
    device,
    metrics,
    lr_s,
):
    run_one_batch = train_one_batch if phase == "train" else eval_one_batch
    phase = "test" if phase == "test" else phase

    # Ensure metric_res always exists
    metric_res = None

    pbar = tqdm(data_loader, disable=__name__ != "__main__")
    for idx, data in enumerate(pbar):
        try:
            attn_type = getattr(model, 'attn_type', None)
        except AttributeError:
            attn_type = None
        try:
            model_name = getattr(model, 'model_name', None)
        except AttributeError:
            model_name = None
        
        if phase == "train" and (attn_type == "None" or model_name== "None"):
            torch.cuda.empty_cache()

        data = process_data(data, phase, device, epoch)

        try:
            batch_loss, batch_embeddings, batch_cluster_ids = run_one_batch(
                model, optimizer, criterion, data, lr_s
            )

            # skip NaN losses
            if math.isnan(batch_loss):
                print("Warning: batch_loss is NaN, skipping this batch")
                continue

            batch_acc, batch_recall = update_metrics(
                metrics, data, batch_embeddings, batch_cluster_ids, criterion.dist_metric
            )
            metrics["loss"].update(batch_loss)

        except RuntimeError as e:
            if "Encountered `nan` values in tensor" in str(e):
                print("Skipping batch due to NaN in batch loss or metrics")
                continue
            else:
                raise

        # build description on every batch
        desc = (
            f"[Epoch {epoch}] {phase}, "
            f"loss: {batch_loss:.4f}, "
            f"acc: {batch_acc:.4f}, "
            f"recall: {batch_recall:.4f}"
        )

        # if this is the last batch, compute epoch‐level metrics
        if idx == len(data_loader) - 1:
            metric_res = compute_metrics(metrics)
            loss, acc = metric_res["loss"], metric_res["accuracy@0.9"]
            prec, recall = metric_res["precision@0.9"], metric_res["recall@0.9"]
            desc = (
                f"[Epoch {epoch}] {phase}, "
                f"loss: {loss:.4f}, "
                f"acc: {acc:.4f}, "
                f"prec: {prec:.4f}, "
                f"recall: {recall:.4f}"
            )
            reset_metrics(metrics)

        pbar.set_description(desc)

    # Fallback: if no batches ran (or all were skipped), compute metrics now
    if metric_res is None:
        metric_res = compute_metrics(metrics)
        reset_metrics(metrics)

    return metric_res


def reset_metrics(metrics):
    for metric in metrics.values():
        if isinstance(metric, MeanMetric):
            metric.reset()


def compute_metrics(metrics):
    return {
        f"{name}@{pt}": metrics[f"{name}@{pt}"].compute().item()
        for name in ["accuracy", "precision", "recall"]
        for pt in metrics["pt_thres"]
    } | {"loss": metrics["loss"].compute().item()}


def update_metrics(metrics, data, batch_embeddings, batch_cluster_ids, dist_metric):
    embeddings = unbatch(batch_embeddings, data.batch)
    cluster_ids = unbatch(batch_cluster_ids, data.batch)
    #acc_09 = 0
    for pt in metrics["pt_thres"]:
        batch_mask = point_filter(batch_cluster_ids, data.reconstructable, data.pt, pt_thres=pt)
        mask = unbatch(batch_mask, data.batch)

        res = [acc_and_pr_at_k(embeddings[i], cluster_ids[i], mask[i], dist_metric) for i in range(len(embeddings))]
        res = torch.tensor(res)
        if torch.isnan(res).any():
            print(f"Skipping update due to NaN values in results for pt={pt}")
            continue
        
        
        metrics[f"accuracy@{pt}"].update(res[:, 0])
        metrics[f"precision@{pt}"].update(res[:, 1])
        metrics[f"recall@{pt}"].update(res[:, 2])
        if pt == 0.9:
            acc_09 = res[:, 0].mean().item()
            recall_09 = res[:, 2].mean().item()
    return acc_09, recall_09


def run_one_seed(config):
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(config["num_threads"])

    dataset_name = config["dataset_name"]
    model_name = config["model_name"]
    dataset_dir = Path(config["data_dir"]) / dataset_name.split("-")[0]
    log(f"Device: {device}, Model: {model_name}, Dataset: {dataset_name}, Note: {config['note']}")

    time = datetime.now().strftime("%m_%d-%H_%M_%S.%f")[:-4]
    rand_num = np.random.randint(10, 100)
    #log_dir = dataset_dir / "logs" / f"{time}{rand_num}_{model_name}_{config['seed']}_{config['note']}"
    log_dir = Path(f"logs/{time}{rand_num}_{model_name}_{config['seed']}_{config['note']}")
    log(f"Log dir: {log_dir}")
    log_dir.mkdir(parents=True, exist_ok=False)
    
    logger = setup_logger(log_dir)
    
    writer = SummaryWriter(log_dir) if config["log_tensorboard"] else None

    set_seed(config["seed"])
    dataset = get_dataset(dataset_name, dataset_dir)
    loaders = get_data_loader(dataset, dataset.idx_split, batch_size=config["batch_size"])

    model = get_model(model_name, config["model_kwargs"], dataset)
    if config.get("only_flops", False):
        raise RuntimeError
    if config.get("resume", False):
        log(f"Resume from {config['resume']}")
        model_path = dataset_dir / "logs" / (config["resume"] + "/best_model.pt")
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint, strict=True)
    model = model.to(device)

    opt = get_optimizer(model.parameters(), config["optimizer_name"], config["optimizer_kwargs"])
    config["lr_scheduler_kwargs"]["num_training_steps"] = config["num_epochs"] * len(loaders["train"])
    lr_s = get_lr_scheduler(opt, config["lr_scheduler_name"], config["lr_scheduler_kwargs"])
    criterion = get_loss(config["loss_name"], config["loss_kwargs"])

    main_metric = config["main_metric"]
    pt_thres = [0, 0.5, 0.9]
    metric_names = ["accuracy", "precision", "recall"]
    print(model_name)
    if model_name == "rwkv7_rwkv7":
        print('rwkv777')
        metrics = {f"{name}@{pt}": MeanMetric(nan_strategy="ignore") for name in metric_names for pt in pt_thres}
        metrics["loss"] = MeanMetric(nan_strategy="ignore")
    else:
        metrics = {f"{name}@{pt}": MeanMetric(nan_strategy="error") for name in metric_names for pt in pt_thres}
        metrics["loss"] = MeanMetric(nan_strategy="error")
    
    metrics["pt_thres"] = pt_thres

    coef = 1 if config["mode"] == "max" else -1
    best_epoch, best_train = 0, {metric: -coef * float("inf") for metric in metrics.keys()}
    best_valid, best_test = deepcopy(best_train), deepcopy(best_train)

    if writer is not None:
        layout = {
            "Gap": {
                "loss": ["Multiline", ["train/loss", "valid/loss", "test/loss"]],
                "acc@0.9": ["Multiline", ["train/accuracy@0.9", "valid/accuracy@0.9", "test/accuracy@0.9"]],
            }
        }
        writer.add_custom_scalars(layout)

    for epoch in range(config["num_epochs"]):
        if not config.get("only_eval", False):
            train_res = run_one_epoch(model, opt, criterion, loaders["train"], "train", epoch, device, metrics, lr_s)
        valid_res = run_one_epoch(model, opt, criterion, loaders["valid"], "valid", epoch, device, metrics, lr_s)
        test_res = run_one_epoch(model, opt, criterion, loaders["test"], "test", epoch, device, metrics, lr_s)

        if lr_s is not None:
            if isinstance(lr_s, ReduceLROnPlateau):
                lr_s.step(valid_res[config["lr_scheduler_metric"]])
            elif isinstance(lr_s, StepLR):
                lr_s.step()

        if (valid_res[main_metric] * coef) > (best_valid[main_metric] * coef):
            best_epoch, best_train, best_valid, best_test = epoch, train_res, valid_res, test_res
            torch.save(model.state_dict(), log_dir / "best_model.pt")

        #print(
        #    f"[Epoch {epoch}] Best epoch: {best_epoch}, train: {best_train[main_metric]:.4f}, "
        #    f"valid: {best_valid[main_metric]:.4f}, test: {best_test[main_metric]:.4f}"
        #)
        
        log_msg = (
            f"[Epoch {epoch}] "
            f"train_loss: {train_res['loss']:.4f}, train_acc@0.9: {train_res['accuracy@0.9']:.4f}, train_recall@0.9: {train_res['recall@0.9']:.4f}, "
            f"valid_loss: {valid_res['loss']:.4f}, valid_acc@0.9: {valid_res['accuracy@0.9']:.4f}, valid_recall@0.9: {valid_res['recall@0.9']:.4f}, "
            f"test_loss: {test_res['loss']:.4f}, test_acc@0.9: {test_res['accuracy@0.9']:.4f}, test_recall@0.9: {test_res['recall@0.9']:.4f}"
        )
        logger.info(log_msg)  # Log the message to both console and file

        print("=" * 50), print("=" * 50)

        if writer is not None:
            writer.add_scalar("lr", opt.param_groups[0]["lr"], epoch)
            for phase, res in zip(["train", "valid", "test"], [train_res, valid_res, test_res]):
                for k, v in res.items():
                    writer.add_scalar(f"{phase}/{k}", v, epoch)
            for phase, res in zip(["train", "valid", "test"], [best_train, best_valid, best_test]):
                for k, v in res.items():
                    writer.add_scalar(f"best_{phase}/{k}", v, epoch)

def main():
    parser = argparse.ArgumentParser(description="Train a model for tracking.")
    parser.add_argument("-m", "--model", type=str, default="hept")
    args = parser.parse_args()

    if args.model in [ "dgcnn", "gravnet"]:
        config_dir = Path(f"./configs/tracking/tracking_gnn_{args.model}.yaml")
    elif args.model in ["rwkv","rwkv7"]:
        config_dir = Path(f"./configs/tracking/tracking_rwkv_{args.model}.yaml")
    elif args.model in ["hmambav1","hmambav2","hydra","gatedelta","fullmamba2","pemamba2","fullhybrid2","fullfullhybrid2","gdlocal1","lshgd"]:
        config_dir = Path(f"./configs/tracking/tracking_mamba_{args.model}.yaml")
    else:
        config_dir = Path(f"./configs/tracking/tracking_trans_{args.model}.yaml")
    config = yaml.safe_load(config_dir.open("r").read())
    run_one_seed(config)


if __name__ == "__main__":
    main()
