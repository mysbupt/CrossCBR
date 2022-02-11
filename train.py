#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
import json
import argparse
from tqdm import tqdm
from itertools import product
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.optim as optim
from utility import Datasets
from models.CrossCBR import CrossCBR


def get_cmd():
    parser = argparse.ArgumentParser()
    # experimental settings
    parser.add_argument("-g", "--gpu", default="0", type=str, help="which gpu to use")
    parser.add_argument("-d", "--dataset", default="Youshu", type=str, help="which dataset to use, options: NetEase, Youshu, iFashion")
    parser.add_argument("-m", "--model", default="CrossCBR", type=str, help="which model to use, options: CrossCBR")
    parser.add_argument("-i", "--info", default="", type=str, help="any auxilary info that will be appended to the log file name")
    args = parser.parse_args()

    return args


def main():
    conf = yaml.safe_load(open("./config.yaml"))
    print("load config file done!")

    paras = get_cmd().__dict__
    dataset_name = paras["dataset"]

    assert paras["model"] in ["CrossCBR"], "Pls select models from: CrossCBR"

    if "_" in dataset_name:
        conf = conf[dataset_name.split("_")[0]]
    else:
        conf = conf[dataset_name]
    conf["dataset"] = dataset_name
    conf["model"] = paras["model"]
    dataset = Datasets(conf)

    conf["gpu"] = paras["gpu"]
    conf["info"] = paras["info"]

    conf["num_users"] = dataset.num_users
    conf["num_bundles"] = dataset.num_bundles
    conf["num_items"] = dataset.num_items

    os.environ['CUDA_VISIBLE_DEVICES'] = conf["gpu"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf["device"] = device
    print(conf)

    for lr, l2_reg, item_level_ratio, bundle_level_ratio, bundle_agg_ratio, embedding_size, num_layers, c_lambda, c_temp in \
            product(conf['lrs'], conf['l2_regs'], conf['item_level_ratios'], conf['bundle_level_ratios'], conf['bundle_agg_ratios'], conf["embedding_sizes"], conf["num_layerss"], conf["c_lambdas"], conf["c_temps"]):
        log_path = "./log/%s/%s" %(conf["dataset"], conf["model"])
        run_path = "./runs/%s/%s" %(conf["dataset"], conf["model"])
        checkpoint_model_path = "./checkpoints/%s/%s/model" %(conf["dataset"], conf["model"])
        checkpoint_conf_path = "./checkpoints/%s/%s/conf" %(conf["dataset"], conf["model"])
        if not os.path.isdir(run_path):
            os.makedirs(run_path)
        if not os.path.isdir(log_path):
            os.makedirs(log_path)
        if not os.path.isdir(checkpoint_model_path):
            os.makedirs(checkpoint_model_path)
        if not os.path.isdir(checkpoint_conf_path):
            os.makedirs(checkpoint_conf_path)

        conf["l2_reg"] = l2_reg
        conf["embedding_size"] = embedding_size

        settings = []
        if conf["info"] != "":
            settings += [conf["info"]]

        settings += [conf["aug_type"]]
        if conf["aug_type"] == "ED":
            settings += [str(conf["ed_interval"])]
        if conf["aug_type"] == "OP":
            assert item_level_ratio == 0 and bundle_level_ratio == 0 and bundle_agg_ratio == 0

        settings += ["Neg_%d" %(conf["neg_num"]), str(conf["batch_size_train"]), str(lr), str(l2_reg), str(embedding_size)]

        conf["item_level_ratio"] = item_level_ratio
        conf["bundle_level_ratio"] = bundle_level_ratio
        conf["bundle_agg_ratio"] = bundle_agg_ratio
        conf["num_layers"] = num_layers
        settings += [str(item_level_ratio), str(bundle_level_ratio), str(bundle_agg_ratio), str(num_layers)]

        conf["c_lambda"] = c_lambda
        conf["c_temp"] = c_temp
        settings += [str(c_lambda), str(c_temp)]

        setting = "_".join(settings)
        log_path = log_path + "/" + setting
        run_path = run_path + "/" + setting
        checkpoint_model_path = checkpoint_model_path + "/" + setting
        checkpoint_conf_path = checkpoint_conf_path + "/" + setting
            
        run = SummaryWriter(run_path)

        # model
        if conf['model'] == 'CrossCBR':
            model = CrossCBR(conf, dataset.graphs).to(device)
        else:
            raise ValueError("Unimplemented model %s" %(conf["model"]))

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=conf["l2_reg"])

        batch_cnt = len(dataset.train_loader)
        test_interval_bs = int(batch_cnt * conf["test_interval"])
        ed_interval_bs = int(batch_cnt * conf["ed_interval"])

        best_metrics, best_perform = init_best_metrics(conf)
        best_epoch = 0
        for epoch in range(conf['epochs']):
            epoch_anchor = epoch * batch_cnt
            model.train(True)
            pbar = tqdm(enumerate(dataset.train_loader), total=len(dataset.train_loader))

            for batch_i, batch in pbar:
                model.train(True)
                optimizer.zero_grad()
                batch = [x.to(device) for x in batch]
                batch_anchor = epoch_anchor + batch_i

                ED_drop = False
                if conf["aug_type"] == "ED" and (batch_anchor+1) % ed_interval_bs == 0:
                    ED_drop = True
                bpr_loss, c_loss = model(batch, ED_drop=ED_drop)
                loss = bpr_loss + conf["c_lambda"] * c_loss
                loss.backward()
                optimizer.step()

                loss_scalar = loss.detach()
                bpr_loss_scalar = bpr_loss.detach()
                c_loss_scalar = c_loss.detach()
                run.add_scalar("loss_bpr", bpr_loss_scalar, batch_anchor)
                run.add_scalar("loss_c", c_loss_scalar, batch_anchor)
                run.add_scalar("loss", loss_scalar, batch_anchor)

                pbar.set_description("epoch: %d, loss: %.4f, bpr_loss: %.4f, c_loss: %.4f" %(epoch, loss_scalar, bpr_loss_scalar, c_loss_scalar))

                if (batch_anchor+1) % test_interval_bs == 0:  
                    metrics = {}
                    metrics["val"] = test(model, dataset.val_loader, conf)
                    metrics["test"] = test(model, dataset.test_loader, conf)
                    best_metrics, best_perform, best_epoch = log_metrics(conf, model, metrics, run, log_path, checkpoint_model_path, checkpoint_conf_path, epoch, batch_anchor, best_metrics, best_perform, best_epoch)


def init_best_metrics(conf):
    best_metrics = {}
    best_metrics["val"] = {}
    best_metrics["test"] = {}
    for key in best_metrics:
        best_metrics[key]["recall"] = {}
        best_metrics[key]["ndcg"] = {}
    for topk in conf['topk']:
        for key, res in best_metrics.items():
            for metric in res:
                best_metrics[key][metric][topk] = 0
    best_perform = {}
    best_perform["val"] = {}
    best_perform["test"] = {}

    return best_metrics, best_perform


def write_log(run, log_path, topk, step, metrics):
    curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    val_scores = metrics["val"]
    test_scores = metrics["test"]

    for m, val_score in val_scores.items():
        test_score = test_scores[m]
        run.add_scalar("%s_%d/Val" %(m, topk), val_score[topk], step)
        run.add_scalar("%s_%d/Test" %(m, topk), test_score[topk], step)

    val_str = "%s, Top_%d, Val:  recall: %f, ndcg: %f" %(curr_time, topk, val_scores["recall"][topk], val_scores["ndcg"][topk])
    test_str = "%s, Top_%d, Test: recall: %f, ndcg: %f" %(curr_time, topk, test_scores["recall"][topk], test_scores["ndcg"][topk])

    log = open(log_path, "a")
    log.write("%s\n" %(val_str))
    log.write("%s\n" %(test_str))
    log.close()

    print(val_str)
    print(test_str)


def log_metrics(conf, model, metrics, run, log_path, checkpoint_model_path, checkpoint_conf_path, epoch, batch_anchor, best_metrics, best_perform, best_epoch):
    for topk in conf["topk"]:
        write_log(run, log_path, topk, batch_anchor, metrics)

    log = open(log_path, "a")

    topk_ = 20
    print("top%d as the final evaluation standard" %(topk_))
    if metrics["val"]["recall"][topk_] > best_metrics["val"]["recall"][topk_] and metrics["val"]["ndcg"][topk_] > best_metrics["val"]["ndcg"][topk_]:
        torch.save(model.state_dict(), checkpoint_model_path)
        dump_conf = dict(conf)
        del dump_conf["device"]
        json.dump(dump_conf, open(checkpoint_conf_path, "w"))
        best_epoch = epoch
        curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for topk in conf['topk']:
            for key, res in best_metrics.items():
                for metric in res:
                    best_metrics[key][metric][topk] = metrics[key][metric][topk]

            best_perform["test"][topk] = "%s, Best in epoch %d, TOP %d: REC_T=%.5f, NDCG_T=%.5f" %(curr_time, best_epoch, topk, best_metrics["test"]["recall"][topk], best_metrics["test"]["ndcg"][topk])
            best_perform["val"][topk] = "%s, Best in epoch %d, TOP %d: REC_V=%.5f, NDCG_V=%.5f" %(curr_time, best_epoch, topk, best_metrics["val"]["recall"][topk], best_metrics["val"]["ndcg"][topk])
            print(best_perform["val"][topk])
            print(best_perform["test"][topk])
            log.write(best_perform["val"][topk] + "\n")
            log.write(best_perform["test"][topk] + "\n")

    log.close()

    return best_metrics, best_perform, best_epoch


def test(model, dataloader, conf):
    tmp_metrics = {}
    for m in ["recall", "ndcg"]:
        tmp_metrics[m] = {}
        for topk in conf["topk"]:
            tmp_metrics[m][topk] = [0, 0]

    device = conf["device"]
    model.eval()
    rs = model.propagate(test=True)
    for users, ground_truth_u_b, train_mask_u_b in dataloader:
        pred_b = model.evaluate(rs, users.to(device))
        pred_b -= 1e8 * train_mask_u_b.to(device)
        tmp_metrics = get_metrics(tmp_metrics, ground_truth_u_b, pred_b, conf["topk"])

    metrics = {}
    for m, topk_res in tmp_metrics.items():
        metrics[m] = {}
        for topk, res in topk_res.items():
            metrics[m][topk] = res[0] / res[1]

    return metrics


def get_metrics(metrics, grd, pred, topks):
    tmp = {"recall": {}, "ndcg": {}}
    for topk in topks:
        _, col_indice = torch.topk(pred, topk)
        row_indice = torch.zeros_like(col_indice) + torch.arange(pred.shape[0], device=pred.device, dtype=torch.long).view(-1, 1)
        is_hit = grd[row_indice.view(-1), col_indice.view(-1)].view(-1, topk)

        tmp["recall"][topk] = get_recall(pred, grd, is_hit, topk)
        tmp["ndcg"][topk] = get_ndcg(pred, grd, is_hit, topk)

    for m, topk_res in tmp.items():
        for topk, res in topk_res.items():
            for i, x in enumerate(res):
                metrics[m][topk][i] += x

    return metrics


def get_recall(pred, grd, is_hit, topk):
    epsilon = 1e-8
    hit_cnt = is_hit.sum(dim=1)
    num_pos = grd.sum(dim=1)

    # remove those test cases who don't have any positive items
    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = (hit_cnt/(num_pos+epsilon)).sum().item()

    return [nomina, denorm]


def get_ndcg(pred, grd, is_hit, topk):
    def DCG(hit, topk, device):
        hit = hit/torch.log2(torch.arange(2, topk+2, device=device, dtype=torch.float))
        return hit.sum(-1)

    def IDCG(num_pos, topk, device):
        hit = torch.zeros(topk, dtype=torch.float)
        hit[:num_pos] = 1
        return DCG(hit, topk, device)

    device = grd.device
    IDCGs = torch.empty(1+topk, dtype=torch.float)
    IDCGs[0] = 1  # avoid 0/0
    for i in range(1, topk+1):
        IDCGs[i] = IDCG(i, topk, device)

    num_pos = grd.sum(dim=1).clamp(0, topk).to(torch.long)
    dcg = DCG(is_hit, topk, device)

    idcg = IDCGs[num_pos]
    ndcg = dcg/idcg.to(device)

    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = ndcg.sum().item()

    return [nomina, denorm]


if __name__ == "__main__":
    main()
