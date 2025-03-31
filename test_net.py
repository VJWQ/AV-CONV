import os
import os.path
import pickle
import json
import torch
import torch.nn as nn
from iopath.common.file_io import PathManager
from sklearn.metrics import average_precision_score, accuracy_score

from model.utils import PerTaskLabel, BinaryAcc, save_checkpoint
from model.ego_exo_model import AVConv
from dataloader import MultiConvDataset

path_manager = PathManager()

def test_net(params):
    # create output directory and log parameters
    logdir = os.path.join(params["out_path"], params["checkpoint_path"].split('/')[-3] + "_inference")
    print("logdir", logdir)
    path_manager.mkdirs(logdir)
    with path_manager.open(os.path.join(logdir, "inference_log.txt"), "w") as f:
        f.write(str(params))
        f.write("\n")

    data_path = params["data"]["data_path"]
    label_path = params["data"]["label_path"]

    # make dataloader
    val_dataset = MultiConvDataset("test", data_path, label_path, params)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params["run_params"]["n_workers"],
        persistent_workers=params["run_params"]["n_workers"] > 0,
        drop_last=False
    )
    print('val_dataloader', len(val_dataloader))

    # construct model and load from checkpoint
    model = AVConv(params).cuda()
    checkpoint = torch.load(params["checkpoint_path"])
    print("=" * 10, "Loading ckpt from {}".format(params["checkpoint_path"]), "=" * 10)
    weights = {k.replace('module.', ''): v for k, v in checkpoint["model_state_dict"].items()}
    model.load_state_dict(weights)
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    # run inference loop
    with torch.no_grad():
        total_steps = len(val_dataloader)
        total_loss = 0.0
        sample_count = 0

        ego_bi_pred = [[], [], [], []]
        ego_bi_true = [[], [], [], []]

        exo_bi_pred = [[], [], [], []]
        exo_bi_true = [[], [], [], []]

        all_metadata = []
        all_ego_pred, all_exo_pred = [], []
        all_ego_label, all_exo_label = [], []
        all_ego_mask, all_exo_mask = [], []

        for idx, data in enumerate(val_dataloader):
            visual_input, audio_input, mask_input, head_bboxes, ego_rels, exo_rels, metadata = data
            visual_input, audio_input, mask_input, head_bboxes, ego_rels, exo_rels = (
                visual_input.cuda(), 
                audio_input.cuda(), 
                mask_input.cuda(),
                head_bboxes.cuda(),
                [l.cuda() for l in ego_rels],
                [l.cuda() for l in exo_rels],
            )

            num_samples = head_bboxes.shape[0]

            ego_edge_label, ego_mask = ego_rels
            exo_edge_label, exo_mask = exo_rels

            ego_label, ego_mask = PerTaskLabel(ego_rels)
            exo_label, exo_mask = PerTaskLabel(exo_rels)

            ego_spk_label, sub_spk_label, ego_lst_label, sub_lst_label = ego_label
            id1_spk_label, id2_spk_label, id1_lst_label, id2_lst_label = exo_label

            ego_spk_mask , sub_spk_mask, ego_lst_mask, sub_lst_mask = ego_mask
            id1_spk_mask, id2_spk_mask, id1_lst_mask, id2_lst_mask = exo_mask

            ego_pred, exo_pred = model(visual_input, audio_input, mask_input, head_bboxes)

            ego_output, exo_output = ego_pred, exo_pred
            
            ego_bi_pred, ego_bi_true, ego_accs = BinaryAcc(ego_pred, ego_label, ego_mask, ego_bi_pred, ego_bi_true)
            exo_bi_pred, exo_bi_true, exo_accs = BinaryAcc(exo_pred, exo_label, exo_mask, exo_bi_pred, exo_bi_true)

            for k in range(len(ego_pred)):
                ego_output[k] = torch.argmax(ego_pred[k], dim=1).tolist()
                exo_output[k] = torch.argmax(exo_pred[k], dim=1).tolist()
                ego_label[k] = ego_label[k].tolist()
                ego_label[k] = [int(a) for a in ego_label[k]]

                exo_label[k] = exo_label[k].tolist()
                ego_mask[k] = ego_mask[k].tolist()
                exo_mask[k] = exo_mask[k].tolist()

            all_metadata.append(metadata)
            all_ego_pred.append(ego_output)
            all_exo_pred.append(exo_output)
            all_ego_label.append(ego_label)
            all_exo_label.append(exo_label)
            all_ego_mask.append(ego_mask)
            all_exo_mask.append(exo_mask)

            sample_count += num_samples

            if idx % 100 == 0:
                print("TEST [%d/%d] " % (idx, total_steps))

                print("~" * 100)

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
        print(" -------- test average acc -------- ", "Ego_acc", ego_acc, "Exo_acc", exo_acc)

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
        print(" -------- test average map -------- ", "Ego_map", ego_map, "Exo_map", exo_map)

        with path_manager.open(os.path.join(logdir, "inference_log.txt"), "a") as f:
            f.write("EGO_SPK_ACC: {:.4f}, SUB_SPK_ACC: {:.4f}, EGO_LST_ACC: {:.4f}, SUB_LST_ACC: {:.4f}\n".format(acc1, acc2, acc3, acc4))
            f.write("ID1_SPK_ACC: {:.4f}, ID2_SPK_ACC: {:.4f}, ID1_LST_ACC: {:.4f}, ID2_LST_ACC: {:.4f}\n".format(acc5, acc6, acc7, acc8))
            f.write("EGO_SPK_MAP: {:.4f}, SUB_SPK_MAP: {:.4f}, EGO_LST_MAP: {:.4f}, SUB_LST_MAP: {:.4f}\n".format(map1, map2, map3, map4))
            f.write("ID1_SPK_MAP: {:.4f}, ID2_SPK_MAP: {:.4f}, ID1_LST_MAP: {:.4f}, ID2_LST_MAP: {:.4f}\n".format(map5, map6, map7, map8))
        file_path = 'preds.pkl'
        print("instances number:", len(all_metadata))
        print(os.path.join(logdir, "inference_log.txt"))
        print("======== saving pkl to logdir", os.path.join(logdir, file_path), " ========")
        with path_manager.open(os.path.join(logdir, file_path), "wb") as file:
            pickle.dump([all_metadata, all_ego_pred, all_exo_pred, all_ego_label, all_exo_label, all_ego_mask, all_exo_mask], file)
        print(f'Predictions saved to {file_path}')

    return 0


if __name__ == '__main__':
    
    params = "./params/params_test.json"
    params = json.loads(open(params).read())["params"]

    test_net(params)

