import os
import os.path
import random
from datetime import datetime
import json
import torch
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from iopath.common.file_io import PathManager
from sklearn.metrics import average_precision_score, accuracy_score

from model.utils import PerTaskLabel, BinaryAcc, save_checkpoint, AverageMeter
from model.ego_exo_model import AVConv
from dataloader import MultiConvDataset


path_manager = PathManager()

# Setup log directory and create tensorboard writer
def setup_logging(params):
    experiment_logdir = os.path.join(
        params["log_path"], datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    path_manager.mkdirs(experiment_logdir)
    path_manager.mkdirs(os.path.join(experiment_logdir, "checkpoints"))

    with path_manager.open(os.path.join(experiment_logdir, "log.txt"), "w") as f:
        f.write(str(params))
        f.write("\n")

    writer = SummaryWriter(log_dir=experiment_logdir)

    return experiment_logdir, writer


def train_net(params):
    logdir, writer = setup_logging(params)
    log_step = 1
    RANDOM_SEED = 8751
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # construct dataloader
    data_path = params["data"]["data_path"]
    label_path = params["data"]["label_path"]

    train_dataset = MultiConvDataset("train", data_path, label_path, params)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["run_params"]["n_workers"],
        persistent_workers=params["run_params"]["n_workers"] > 0,
        drop_last=True
    )
    val_dataset = MultiConvDataset("test", data_path, label_path, params)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=params["run_params"]["n_workers"],
        persistent_workers=params["run_params"]["n_workers"] > 0,
        drop_last=False
    )

    # construct model
    model = AVConv(params).cuda()
    model = torch.nn.DataParallel(model, device_ids=list(range(0, params["run_params"]["n_gpu"])))

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), params["lr"])
    loss_fn = nn.CrossEntropyLoss()
    
    # main training loop
    global_step = 0
    print('train_dataloader', len(train_dataloader))
    print('val_dataloader', len(val_dataloader))

    for epoch in range(params["max_epochs"]):
        total_steps = len(train_dataloader)
        loss_meter = AverageMeter()

        ego_bi_pred = [[], [], [], []]
        ego_bi_true = [[], [], [], []]

        exo_bi_pred = [[], [], [], []]
        exo_bi_true = [[], [], [], []]

        for idx, data in enumerate(train_dataloader):
            visual_input, audio_input, mask_input, head_bboxes, ego_rels, exo_rels, _ = data
            visual_input, audio_input, mask_input, head_bboxes, ego_rels, exo_rels = (
                visual_input.cuda(), 
                audio_input.cuda(), 
                mask_input.cuda(),
                head_bboxes.cuda(),
                [l.cuda() for l in ego_rels],
                [l.cuda() for l in exo_rels],
            )
            optimizer.zero_grad()
            ego_edge_label, ego_mask = ego_rels
            exo_edge_label, exo_mask = exo_rels

            ego_label, ego_mask = PerTaskLabel(ego_rels)
            exo_label, exo_mask = PerTaskLabel(exo_rels)

            ego_spk_label, sub_spk_label, ego_lst_label, sub_lst_label = ego_label
            id1_spk_label, id2_spk_label, id1_lst_label, id2_lst_label = exo_label

            ego_spk_mask , sub_spk_mask, ego_lst_mask, sub_lst_mask = ego_mask
            id1_spk_mask, id2_spk_mask, id1_lst_mask, id2_lst_mask = exo_mask

            # # Model analysis
            # from nni.compression.utils.counter import count_flops_params
            # flops, params, _ = count_flops_params(model, (inputs[0][0], inputs[1][0], inputs[2][0], head_bboxes))
            # # Count the number of parameters
            # num_params = sum(p.numel() for p in model.parameters())
            # print(f"Number of parameters in the model: {num_params}")
            # gflops = flops / (10**9)
            # print(f"Total FLOPs: {flops}")
            # print(f"Total GFLOPs: {gflops:.2f}G", f"Total Params: {num_params}")

            ego_pred, exo_pred = model(visual_input, audio_input, mask_input, head_bboxes)

            ego_spk, sub_spk, ego_lst, sub_lst = ego_pred
            id1_spk, id2_spk, id1_lst, id2_lst = exo_pred

            # logging losses: 
            loss1 = loss_fn(ego_spk, ego_spk_label.long())
            loss2 = loss_fn(sub_spk, sub_spk_label.long())
            loss3 = loss_fn(ego_lst, ego_lst_label.long())
            loss4 = loss_fn(sub_lst, sub_lst_label.long())

            loss5 = loss_fn(id1_spk, id1_spk_label.long())
            loss6 = loss_fn(id2_spk, id2_spk_label.long())
            loss7 = loss_fn(id1_lst, id1_lst_label.long())
            loss8 = loss_fn(id2_lst, id2_lst_label.long())

            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8

            # converting preds/gts
            ego_bi_pred, ego_bi_true, ego_accs = BinaryAcc(ego_pred, ego_label, ego_mask, ego_bi_pred, ego_bi_true)
            exo_bi_pred, exo_bi_true, exo_accs = BinaryAcc(exo_pred, exo_label, exo_mask, exo_bi_pred, exo_bi_true)

            loss.backward()
            optimizer.step()

            if idx % log_step == 0:
                print(
                    "TRAIN [%d, %d/%d, %d] loss: %f"
                    % (epoch, idx, total_steps, global_step, loss.item())
                )

            global_step += 1
            if writer is not None:
                writer.add_scalar("Iter/iter_loss", loss, global_step=global_step)
                loss_meter.update(loss, len(data[0]))

        
        cs_pred, ss_pred, cl_pred, sl_pred = ego_bi_pred
        cs_true, ss_true, cl_true, sl_true = ego_bi_true

        ls_pred, rs_pred, ll_pred, rl_pred = exo_bi_pred
        ls_true, rs_true, ll_true, rl_true = exo_bi_true

        # Acc evaluation
        accuracy1 = accuracy_score(cs_true, cs_pred)
        accuracy2 = accuracy_score(ss_true, ss_pred)
        accuracy3 = accuracy_score(cl_true, cl_pred)
        accuracy4 = accuracy_score(sl_true, sl_pred)

        accuracy5 = accuracy_score(ls_true, ls_pred)
        accuracy6 = accuracy_score(rs_true, rs_pred)
        accuracy7 = accuracy_score(ll_true, ll_pred)
        accuracy8 = accuracy_score(rl_true, rl_pred)

        ego_acc = (accuracy1 + accuracy2 + accuracy3 + accuracy4) / 4
        exo_acc = (accuracy5 + accuracy6 + accuracy7 + accuracy8) / 4

        # mAP evaluation
        map1 = average_precision_score(cs_true, cs_pred)
        map2 = average_precision_score(ss_true, ss_pred)
        map3 = average_precision_score(cl_true, cl_pred)
        map4 = average_precision_score(sl_true, sl_pred)

        map5 = average_precision_score(ls_true, ls_pred)
        map6 = average_precision_score(rs_true, rs_pred)
        map7 = average_precision_score(ll_true, ll_pred)
        map8 = average_precision_score(rl_true, rl_pred)
        ego_map = (map1 + map2 + map3 + map4) / 4
        exo_map = (map5 + map6 + map7 + map8) / 4

        avg_loss = loss_meter.avg
        print(
            "TRAIN EPOCH [%d] loss: %04f ego acc: %04f exo acc: %04f ego map: %04f exo map: %04f"
            % (epoch, avg_loss, ego_acc, exo_acc, ego_map, exo_map)
        )

        print('-' * 100)
        if writer is not None:
            writer.add_scalar("Train/epoch_loss", avg_loss, global_step=epoch)
            writer.add_scalar("Train/1-ego_acc", ego_acc, global_step=epoch)
            writer.add_scalar("Train/2-exo_acc", exo_acc, global_step=epoch)
            writer.add_scalar("Train/3-ego_map", ego_map, global_step=epoch)
            writer.add_scalar("Train/4-exo_map", exo_map, global_step=epoch)

        # save ckpt
        save_checkpoint(
            model,
            optimizer,
            logdir,
            epoch,
            global_step,
        )

        # val epoch
        with torch.no_grad():
            model.eval()
            total_steps = len(val_dataloader)
            total_loss = 0.0
            sample_count = 0
            val_loss_meter = AverageMeter()

            ego_bi_pred = [[], [], [], []]
            ego_bi_true = [[], [], [], []]

            exo_bi_pred = [[], [], [], []]
            exo_bi_true = [[], [], [], []]

            for idx, data in enumerate(val_dataloader):
                visual_input, audio_input, mask_input, head_bboxes, ego_rels, exo_rels, _ = data
                visual_input, audio_input, mask_input, head_bboxes, ego_rels, exo_rels = (
                    visual_input.cuda(), 
                    audio_input.cuda(), 
                    mask_input.cuda(),
                    head_bboxes.cuda(),
                    [l.cuda() for l in ego_rels],
                    [l.cuda() for l in exo_rels],
                )

                ego_edge_label, ego_mask = ego_rels
                exo_edge_label, exo_mask = exo_rels

                ego_label, ego_mask = PerTaskLabel(ego_rels)
                exo_label, exo_mask = PerTaskLabel(exo_rels)

                ego_spk_label, sub_spk_label, ego_lst_label, sub_lst_label = ego_label
                id1_spk_label, id2_spk_label, id1_lst_label, id2_lst_label = exo_label

                ego_spk_mask , sub_spk_mask, ego_lst_mask, sub_lst_mask = ego_mask
                id1_spk_mask, id2_spk_mask, id1_lst_mask, id2_lst_mask = exo_mask

                ego_pred, exo_pred = model(visual_input, audio_input, mask_input, head_bboxes)

                ego_spk, sub_spk, ego_lst, sub_lst = ego_pred
                id1_spk, id2_spk, id1_lst, id2_lst = exo_pred

                loss1 = loss_fn(ego_spk, ego_spk_label.long())
                loss2 = loss_fn(sub_spk, sub_spk_label.long())
                loss3 = loss_fn(ego_lst, ego_lst_label.long())
                loss4 = loss_fn(sub_lst, sub_lst_label.long())

                loss5 = loss_fn(id1_spk, id1_spk_label.long())
                loss6 = loss_fn(id2_spk, id2_spk_label.long())
                loss7 = loss_fn(id1_lst, id1_lst_label.long())
                loss8 = loss_fn(id2_lst, id2_lst_label.long())

                loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8

                # converting preds/gts
                ego_bi_pred, ego_bi_true, ego_accs = BinaryAcc(ego_pred, ego_label, ego_mask, ego_bi_pred, ego_bi_true)
                exo_bi_pred, exo_bi_true, exo_accs = BinaryAcc(exo_pred, exo_label, exo_mask, exo_bi_pred, exo_bi_true)

                if idx % log_step == 0:
                    print(
                        "VAL [%d, %d/%d, %d] loss: %f"
                        % (epoch, idx, total_steps, global_step, loss.item())
                    )
                    val_loss_meter.update(loss, len(data[0]))

            cs_pred, ss_pred, cl_pred, sl_pred = ego_bi_pred
            cs_true, ss_true, cl_true, sl_true = ego_bi_true

            ls_pred, rs_pred, ll_pred, rl_pred = exo_bi_pred
            ls_true, rs_true, ll_true, rl_true = exo_bi_true

            # Acc evaluation
            acc1 = accuracy_score(cs_true, cs_pred)
            acc2 = accuracy_score(ss_true, ss_pred)
            acc3 = accuracy_score(cl_true, cl_pred)
            acc4 = accuracy_score(sl_true, sl_pred)

            acc5 = accuracy_score(ls_true, ls_pred)
            acc6 = accuracy_score(rs_true, rs_pred)
            acc7 = accuracy_score(ll_true, ll_pred)
            acc8 = accuracy_score(rl_true, rl_pred)

            ego_acc = (acc1 + acc2 + acc3 + acc4) / 4
            exo_acc = (acc5 + acc6 + acc7 + acc8) / 4

            # mAP evaluation
            map1 = average_precision_score(cs_true, cs_pred)
            map2 = average_precision_score(ss_true, ss_pred)
            map3 = average_precision_score(cl_true, cl_pred)
            map4 = average_precision_score(sl_true, sl_pred)

            map5 = average_precision_score(ls_true, ls_pred)
            map6 = average_precision_score(rs_true, rs_pred)
            map7 = average_precision_score(ll_true, ll_pred)
            map8 = average_precision_score(rl_true, rl_pred)

            ego_map = (map1 + map2 + map3 + map4) / 4
            exo_map = (map5 + map6 + map7 + map8) / 4
            print("=" * 50)

            avg_loss = val_loss_meter.avg
            print(
                "VAL EPOCH [%d] loss: %04f ego acc: %04f exo acc: %04f ego map: %04f exo map: %04f"
                % (epoch, avg_loss, ego_acc, exo_acc, ego_map, exo_map)
            )

            if writer is not None:
                writer.add_scalar("Val/epoch_loss", avg_loss, global_step=epoch)
                writer.add_scalar("Val/1-ego_spk_acc", acc1, global_step=epoch)
                writer.add_scalar("Val/2-sub_spk_acc", acc2, global_step=epoch)
                writer.add_scalar("Val/3-ego_lst_acc", acc3, global_step=epoch)
                writer.add_scalar("Val/4-sub_lst_acc", acc4, global_step=epoch)
                writer.add_scalar("Val/5-lft_spk_acc", acc5, global_step=epoch)
                writer.add_scalar("Val/6-rgt_spk_acc", acc6, global_step=epoch)
                writer.add_scalar("Val/7-lft_lst_acc", acc7, global_step=epoch)
                writer.add_scalar("Val/8-rgt_lst_acc", acc8, global_step=epoch)
                writer.add_scalar("Val/9-ego_accuracy", ego_acc, global_step=epoch)
                writer.add_scalar("Val/10-exo_accuracy", exo_acc, global_step=epoch)

                writer.add_scalar("Val/11-ego_map", ego_map, global_step=epoch)
                writer.add_scalar("Val/12-exo_map", exo_map, global_step=epoch)
                writer.add_scalar("Val/13-ego_spk_map", map1, global_step=epoch)
                writer.add_scalar("Val/14-sub_spk_map", map2, global_step=epoch)
                writer.add_scalar("Val/15-ego_lst_map", map3, global_step=epoch)
                writer.add_scalar("Val/16-sub_lst_map", map4, global_step=epoch)
                writer.add_scalar("Val/17-lft_spk_map", map5, global_step=epoch)
                writer.add_scalar("Val/18-rgt_spk_map", map6, global_step=epoch)
                writer.add_scalar("Val/19-lft_lst_map", map7, global_step=epoch)
                writer.add_scalar("Val/20-rgt_lst_map", map8, global_step=epoch)

        with path_manager.open(os.path.join(logdir, "log.txt"), "a") as f:
            f.write(
                "Val epoch at epoch {}, global step {} ego acc: {:.4f} exo acc: {:.4f} ego map: {:.4f} exo map: {:.4f} loss: {:.4f}\n".format(
                    epoch,
                    global_step,
                    ego_acc,
                    exo_acc,
                    ego_map,
                    exo_map,
                    avg_loss,
                )
            )

    return 0


if __name__ == '__main__':
    
    params = "./params/params_train.json"
    params = json.loads(open(params).read())["params"]

    train_net(params)

