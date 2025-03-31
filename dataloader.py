import json
import os
import os
import cv2
import numpy as np
import scipy.io as sio
import scipy.signal
import torch
import torchaudio
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", "(?s).*MATPLOTLIBDATA.*", category=UserWarning)

class MultiConvDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, mode, data_dir, label_dir, params):
        self.pad = 4
        self.mode = mode
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.img_size = params["data"]["image_size"]
        self.bbox_scaler = params["data"]["bbox_scale_factor"]
        self.input_size = (self.img_size[0] // self.bbox_scaler, self.img_size[1] // self.bbox_scaler)
        self.num_sub = params["data"]["num_subjects"]
        self.clip_stride = params["data"]["clip_stride"]
        self.visual_num_frames = params["data"]["visual_num_frames"]
        self.visual_sampling_rate = params["data"]["visual_sampling_rate"]
        self.audio_num_frames = params["data"]["audio_num_frames"]
        self.audio_sampling_rate = params["data"]["audio_sampling_rate"]
        self.spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=400, power=None, win_length=20)
        self.window_length = self.visual_num_frames * self.visual_sampling_rate
        self.ini = (self.window_length - 1) % self.visual_sampling_rate  # make sure to include the last frame
        self.session_names = []  # [session_idx] -> session name string
        self.original_frames = []  # [session_idx][frame_idx] -> original frame int
        self.all_ego_edges, self.all_exo_edges = [], []
        self.all_ego_abses, self.all_exo_abses = [], []
        self.all_ego_masks, self.all_exo_masks = [], []
        self.all_heads = []
        self.clip_list = []  # [idx] -> [(session_idx, frame_idx)...]

        if self.mode == "train":
            session_list = [
                path.name
                for path in os.scandir(data_dir)
                if path.name[-5:] == ".json"
                and (int(path.name.split("_")[0]) in [1, 2, 3, 4, 5, 6, 7])
            ]
            session_list = ['1_1_1.json']
        else:
            session_list = [
                path.name
                for path in os.scandir(data_dir)
                if path.name[-5:] == ".json"
                and (int(path.name.split("_")[0]) in [8, 9, 10])
            ]
            session_list = ['1_1_1.json']

        print(session_list)

        for session_idx in range(len(session_list)):
            session_name = session_list[session_idx][0:-5]
            ppl_list = list(range(1, self.num_sub + 2))
            self.wearer = int(session_name[-1:])
            self.sub_ppl_list = ppl_list[:]
            self.sub_ppl_list.remove(int(self.wearer))
            self.possible_set = sorted(set((i, j) for i in self.sub_ppl_list for j in self.sub_ppl_list if i != j and i < j))

            sess_original_frames = []
            sess_norm_ego_edges = []
            sess_norm_exo_edges = []
            sess_norm_heads = []
            sess_ego_abs_idxes = []
            sess_exo_abs_idxes = []
            sess_ego_abs_masks = []
            sess_exo_abs_masks = []

            with open(os.path.join(self.label_dir, session_name + ".json"), "r") as f:
                frames = json.load(f)

            for frame_idx in range(len(frames)):
                frame = frames[frame_idx]

                head_pair = np.array([(frame["pairwise"][i]["subject"]) for i in range(len(frame["pairwise"]))])
                heads = np.array(sorted(frame["bboxes"], key=lambda x: x[4]))
                if head_pair.ndim > 1:
                    head_pair[:, :, 0:4] = head_pair[:, :, 0:4] / float(self.bbox_scaler)
                    heads[:, 0:4] = heads[:, 0:4] / float(self.bbox_scaler)

                ego_spk = frame["wearer_speaking"]
                ego_edge, ego_abs_idx, ego_abs_mask = self.find_directional_ego_rels(heads, ego_spk, self.wearer)
                exo_edge, exo_abs_idx, exo_abs_mask = self.find_directional_exo_rels(head_pair, self.possible_set)

                if len(heads) == 0:
                    normed_heads = np.array([[0 for _ in range(4)] for _ in range(4)])
                else:
                    normed_heads = heads[:,:4]

                    for i in ego_abs_idx:
                        normed_heads = np.insert(normed_heads, i, 0, axis=0)

                head_pair = head_pair.astype(int).tolist()
                normed_heads = normed_heads.astype(int).tolist()

                sess_original_frames.append(frame["frame"])
                sess_norm_heads.append(normed_heads)
                sess_norm_ego_edges.append(list(ego_edge.values()))
                sess_norm_exo_edges.append(list(exo_edge.values()))
                sess_ego_abs_idxes.append(ego_abs_idx)
                sess_exo_abs_idxes.append(exo_abs_idx)
                sess_ego_abs_masks.append(ego_abs_mask)
                sess_exo_abs_masks.append(exo_abs_mask)

                # Finding valid clips
                if (frame_idx + 1) % self.clip_stride == 0:
                    clip_idxs, _ = self.get_clip_idxs(session_idx, frame_idx, len(frames) - 1)
                    self.clip_list.append(clip_idxs)

            self.session_names.append(session_name)
            self.original_frames.append(sess_original_frames)
            self.all_heads.append(sess_norm_heads)
            self.all_ego_edges.append(sess_norm_ego_edges)
            self.all_exo_edges.append(sess_norm_exo_edges)
            self.all_ego_abses.append(sess_ego_abs_idxes)
            self.all_exo_abses.append(sess_exo_abs_idxes)
            self.all_ego_masks.append(sess_ego_abs_masks)
            self.all_exo_masks.append(sess_exo_abs_masks)

        print("{} split: {} clips".format(mode, len(self.clip_list)))

    # gets list of (session_idx, frame_idx) for inputs belong with this particular example
    # (session_idx, frame_idx) denotes the last frame in the example
    def get_clip_idxs(self, session_idx, frame_idx, session_last_idx):
        inputs = []
        duplicate_mask = []
        for i in range(frame_idx - self.window_length + 1, frame_idx + 1):
            # if out of range at end, we need to pad the clip with the last frame in the session
            if i > session_last_idx:
                duplicate_mask.append(1)  # indicate these are padding
                inputs.append((session_idx, session_last_idx))
            else:
                duplicate_mask.append(0)
                inputs.append((session_idx, i))

        return inputs, duplicate_mask

    def find_directional_ego_rels(self, heads, ego_spk, wearer):
        absence = (2,) * self.num_sub
        directional_ego_dict = {tuple([wearer, new_list]): absence for new_list in self.sub_ppl_list}    # key(possible pair):value(relationship label)

        for i in heads:
            pair = [wearer, i[4]]
            sub_spk = i[5]
            grp = int(i[-1])
            ego_lst = int(sub_spk and grp)
            sub_lst = int(ego_spk and grp)
            spk_lis = (ego_spk, sub_spk, ego_lst, sub_lst)
            directional_ego_dict[tuple(pair)] = spk_lis
        absence_sub = [k[1] for k, v in directional_ego_dict.items() if v == absence]
        absence_idx = [self.sub_ppl_list.index(t) for t in absence_sub]
        absence_mask = [[0] * self.num_sub if i in absence_idx else [1] * self.num_sub for i in range(len(directional_ego_dict))]

        return directional_ego_dict, absence_idx, absence_mask

    def find_directional_exo_rels(self, head_pairs, possible_set):
        absence = (2,) * self.num_sub
        directional_exo_dict = {
            new_list: absence for new_list in possible_set
        }
        for i in head_pairs:
            sorted_ids = sorted(i, key=lambda x: x[4])
            p1, p2 = sorted_ids
            id1, id2 = p1[4], p2[4]
            pair = [id1, id2]
            id1_spk, id2_spk = p1[5], p2[5]
            grp = int(p1[-1] == p2[-1])
            id1_lst = int(id2_spk and grp)
            id2_lst = int(id1_spk and grp)
            spk_lis = (id1_spk, id2_spk, id1_lst, id2_lst)
            directional_exo_dict[tuple(pair)] = spk_lis

        absence_sub = [k for k, v in directional_exo_dict.items() if v == absence]
        absence_idx = [possible_set.index(t) for t in absence_sub]
        absence_mask = [[0] * self.num_sub if i in absence_idx else [1] * self.num_sub for i in range(len(directional_exo_dict))]

        return directional_exo_dict, absence_idx, absence_mask

    def get_image_input(self, seq_sess, seq_frmid):
        head_img_inputs = torch.zeros(
            self.visual_num_frames, self.img_size[0], self.img_size[1]
        )
        for i in range(0, self.visual_num_frames):
            img_path = os.path.join(
                self.data_dir,
                seq_sess,
                "image_" + str(seq_frmid[i]) + ".jpg",
            )
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)    # (400, 424)
            ratio = self.img_size[1] / self.img_size[0]
            img = cv2.resize(img, (0,0), fx=ratio, fy=ratio)    # (200, 212)
            butt = self.img_size[0] - img.shape[0]
            img = cv2.copyMakeBorder(img, 0, butt, 0, 0, cv2.BORDER_REPLICATE)
            img = img[:, :self.img_size[1]]
            img = torch.FloatTensor(img) / 255
            head_img_inputs[i] = img

        head_img_inputs = head_img_inputs.unsqueeze(dim=1)  # recover the channel dim
        head_img_inputs = F.interpolate(head_img_inputs, size=self.input_size[0])

        return head_img_inputs

    def get_audio_input(self, seq_sess, seq_frmid):
        audio_inputs = []
        for i in range(0, self.audio_num_frames):
            pid = seq_sess[-1]
            path = os.path.join(
                self.data_dir,
                seq_sess,
                "a"
                + pid
                + "_"
                + str(seq_frmid[i])
                + ".mat",
            )
            aud = sio.loadmat(path)
            aud = aud["aud"].astype("float32")
            # normalize
            for n in range(6):
                s = np.sqrt(np.sum(aud[n, :] * aud[n, :]))
                aud[n, :] = aud[n, :] / (s + 1e-3)

            aim_corr = []
            for n in range(6):
                for m in range(6):
                    if n == m:
                        continue
                    xc = scipy.signal.correlate(aud[n, :], aud[m, :], method="fft")
                    mid = len(xc) // 2
                    if len(aim_corr) == 0:
                        aim_corr = xc[mid - 50 : mid + 50]
                    else:
                        aim_corr = np.vstack((aim_corr, xc[mid - 50 : mid + 50]))
            aim_corr = cv2.resize(aim_corr, self.input_size)
            aim_corr = torch.Tensor(aim_corr)

            aim_real = []
            aim_imag = []
            for idx in range(6):
                spec = self.spectrogram_transform(torch.Tensor(aud[idx, :]))
                aim_real.append(spec.real)
                aim_imag.append(spec.imag)
            aim_real = torch.stack(aim_real)
            aim_real = aim_real.reshape(aim_real.shape[1] * 3, aim_real.shape[2] * 2)
            aim_real = torch.Tensor(cv2.resize(aim_real.numpy(), self.input_size))

            aim_imag = torch.stack(aim_imag)
            aim_imag = aim_imag.reshape(aim_imag.shape[1] * 3, aim_imag.shape[2] * 2)
            aim_imag = torch.Tensor(cv2.resize(aim_imag.numpy(), self.input_size))

            aim = torch.stack([aim_corr, aim_real, aim_imag])
            audio_inputs.append(aim)

        audio_inputs = torch.stack(audio_inputs)

        return audio_inputs

    def get_all_heads_mask(self, seq_heads):
        heads_count = len(seq_heads[0])
        head_msk_inputs = torch.zeros(
            self.visual_num_frames, heads_count, self.img_size[0], self.img_size[1]
        )
        pad = self.pad
        for i in range(0, self.visual_num_frames):
            for j in range(heads_count):
                if sum(seq_heads[i][j]) != 0:
                    x1, y1, x2, y2 = seq_heads[i][j]
                    head_msk_inputs[i][j][y1-pad:y2+pad, x1-pad:x2+pad] = 1

        head_msk_inputs = F.interpolate(head_msk_inputs, size=self.input_size[0])

        return head_msk_inputs

    def __getitem__(self, index):
        seq_frms, seq_heads = [], []
        seq_ego_edges, seq_exo_edges = [], []
        seq_ego_masks, seq_exo_masks = [], []
        seq_sess = ''
        for i in range(0, self.visual_num_frames):
            session_idx, frame_idx = self.clip_list[index][i * self.visual_sampling_rate + self.ini]
            seq_frms.append(str(self.original_frames[session_idx][frame_idx]))
            seq_heads.append(self.all_heads[session_idx][frame_idx])
            seq_ego_edges.append(self.all_ego_edges[session_idx][frame_idx])
            seq_exo_edges.append(self.all_exo_edges[session_idx][frame_idx])
            seq_ego_masks.append(self.all_ego_masks[session_idx][frame_idx])
            seq_exo_masks.append(self.all_exo_masks[session_idx][frame_idx])
        seq_sess = self.session_names[session_idx]

        visual_input = self.get_image_input(seq_sess, seq_frms)
        audio_input = self.get_audio_input(seq_sess, seq_frms)
        mask_input = self.get_all_heads_mask(seq_heads)

        head_bboxes = torch.FloatTensor([seq_heads]).squeeze()

        ego_edge_label = torch.FloatTensor([seq_ego_edges]).squeeze()
        exo_edge_label = torch.FloatTensor([seq_exo_edges]).squeeze()
        ego_masks = torch.FloatTensor([seq_ego_masks]).squeeze()
        exo_masks = torch.FloatTensor([seq_exo_masks]).squeeze()
        ego_rels = [ego_edge_label, ego_masks]
        exo_rels = [exo_edge_label, exo_masks]

        metadata = {
            "session": seq_sess,
            "frames": seq_frms,
        }

        return visual_input, audio_input, mask_input, head_bboxes, ego_rels, exo_rels, metadata

    def __len__(self):
        return len(self.clip_list)
