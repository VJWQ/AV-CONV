import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from models.transformer import Transformer
from torchvision.ops import RoIAlign

class Res18(nn.Module):
    def __init__(self, in_dim):
        super(Res18, self).__init__()
        self.model1 = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        self.conv0 = nn.Conv2d(in_dim, 64, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.model1.bn1(x)
        x = self.model1.relu(x)
        x = self.model1.layer1(x)
        x = self.model1.layer2(x)
        x = self.model1.layer3(x)
        x = self.model1.layer4(x)

        return x

class PredHead(nn.Module):
    def __init__(self, in_dim, temp_conv="vanilla"):
        super(PredHead, self).__init__()
        self.temp_conv = temp_conv
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cnn1  = nn.Conv1d(in_dim, in_dim, 3, padding=1, padding_mode="replicate")
        self.fc1 = nn.Linear(in_features=in_dim, out_features=3, bias=True)
        self.fc2 = nn.Linear(in_features=in_dim // 2, out_features=3, bias=True)
        self.in_proj  = nn.Conv2d(in_dim, in_dim // 2, kernel_size=1)

    def forward(self, x):
        if self.temp_conv == "conv1d":
            x = torch.mean(x, dim=3)
            x = self.cnn1(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = torch.flatten(x, 0, 1)
            x = self.fc1(x)
        else:   # vanilla
            x = torch.mean(x, dim=3)
            x = torch.flatten(x, 0, 1)
            x = self.fc1(x)
        return x

class AVConv(nn.Module):
    def __init__(self, params):
        super(AVConv, self).__init__()
        self.sub_idx_list = [0, 1, 2, 3]
        # self.sub_idx_list = [0, 1, 2, 3, 4]
        self.idx_pair = sorted(
            set((i, j) for i in self.sub_idx_list for j in self.sub_idx_list if i != j and i < j)
        )
        self.hidden_dim = [512, 256, 14, 14]
        self.in_dim = [256, 512, 1024]
        self.in_channels = [1, 3, 4]
        self.viz_encoder_2d = Res18(in_dim=self.in_channels[0])
        self.aud_pos_encoder_2d = Res18(in_dim=self.in_channels[2])
        self.in_proj = nn.Conv2d(self.hidden_dim[0], self.hidden_dim[1], kernel_size=1)
        self.spa_proj = nn.Conv1d(196, 16, 3, padding=1, padding_mode="replicate")
        pred_head = params["model"]["temp_conv"]  # "vanilla", "conv1d"
        self.num_attn = params["model"]["num_attn"]
        self.attn_type = params["model"]["attn_type"]  # T_N_S, TN, TS, NS, T, N, S, None (DIRECT CONCAT)
        self.num_frame = params["data"]["visual_num_frames"]
        self.num_sub = params["data"]["num_subjects"]
        self.low_spa = params["model"]["low_spatial_dim"]     # option to project 196 to 16
        print("=" * 20, "initialing ego_exo", "=" * 20)
        print("self-attn types", self.attn_type, len(self.attn_type.split("_")))

        if self.attn_type != "None":
            self.div_list = ["TS", "T_N_S"]
            embed_dim = 512     # CM = 2 x 256 = 512
            # adding positional embedding on T/S/N
            self.temp_embed = nn.Parameter(torch.zeros(1, self.num_frame, embed_dim))
            self.subj_embed = nn.Parameter(torch.zeros(1, self.num_sub, embed_dim))
            self.spat_embed = nn.Parameter(torch.zeros(1, 196, embed_dim))
            # self.conv_attn = CONV_ATTN(self.num_attn, embed_dim, attn_type=self.attn_type, num_frame=self.num_frame, num_sub=self.num_sub)
            self.conv_attn = Transformer(self.num_attn, embed_dim, attn_type=self.attn_type, num_frame=self.num_frame, num_sub=self.num_sub)

            
            print("self.num_frame", self.num_frame, self.attn_type, "embed_dim", embed_dim)
            print("self-attn types", self.attn_type.split("_"))

        # pairwise self-attention: heavy, not included in AV-CONV final model
        self.pair_attn = params["model"]["pair_attn"]
        if self.pair_attn != "None":
            pair_token_dim, pair_embed_dim = 1176, 1024
            # self.pairwise_attn = CONV_ATTN(self.num_attn, pair_embed_dim, pair_token_dim, attn_type=self.pair_attn, num_frame=self.num_frame, num_sub=6)
            self.pairwise_attn = Transformer(self.num_attn, pair_embed_dim, pair_token_dim, attn_type=self.pair_attn, num_frame=self.num_frame, num_sub=6)

        self.abla = params["model"]["abla"]   # self.abla: if running single-modality experiment or not
        if self.abla:
            ego_dim = self.in_dim[0]
            exo_dim = self.in_dim[1]
        else:
            ego_dim = self.in_dim[1]
            exo_dim = self.in_dim[2]
        # initializing FCN classifiers for each task
        self.cls_ego_spk = PredHead(in_dim=ego_dim, temp_conv=pred_head)
        self.cls_sub_spk = PredHead(in_dim=ego_dim, temp_conv=pred_head)
        self.cls_ego_lst = PredHead(in_dim=ego_dim, temp_conv=pred_head)
        self.cls_sub_lst = PredHead(in_dim=ego_dim, temp_conv=pred_head)

        self.cls_id1_spk = PredHead(in_dim=exo_dim, temp_conv=pred_head)
        self.cls_id2_spk = PredHead(in_dim=exo_dim, temp_conv=pred_head)
        self.cls_id1_lst = PredHead(in_dim=exo_dim, temp_conv=pred_head)
        self.cls_id2_lst = PredHead(in_dim=exo_dim, temp_conv=pred_head)

        output_size = (210, 210)
        spatial_scale = 1 / 2
        self.roi_align = RoIAlign(output_size, spatial_scale=spatial_scale, sampling_ratio=-1)


    def forward(self, visual_input, audio_input, head_pos, normed_roi):
        '''
        -----------------------------------------------
        | B: batch size                               |
        | T: number of temporal frames                |
        | N: numbder of subjects                      |
        | S:                                          |
        -----------------------------------------------
        | img         torch.Size([B, T, 1, 210, 210]) |
        | aud         torch.Size([B, T, 3, 210, 210]) |
        | head_pos    torch.Size([B, T, 4, 210, 210]) |
        | normed_roi  torch.Size([B, T, N, 4])        |
        -----------------------------------------------
        '''

        # Reshape ROIs: [B, T, N, K] -> [B*T*N, K]
        # Extract local features using RoIAlign
        B, T, N, K = normed_roi.shape
        normed_roi = normed_roi.reshape(B * T * N, K)
        batch_idx = torch.arange(B * T).unsqueeze(1).T
        batch_idx_list = batch_idx.repeat(N, 1).T.reshape(B * T * N, 1).cuda()
        normed_roi = torch.cat((batch_idx_list, normed_roi), dim=1)     # [B*T*N, K]

        feat_list = []
        if len(visual_input):
            img = visual_input       
            B, T, C, H, W = img.shape           # [B, T, 1, 210, 210]
            img = img.reshape(B * T, C, H, W)   # [B*T, 1, 210, 210]

            # 1. Prepare head imput: [B*T, 1, 210, 210] -> [B*T*N, 1, 210, 210]
            head_img = self.roi_align(img, normed_roi)

            # Extract image feature: [B*T*N, 1, 210, 210] -> [B*T*N, 512, 14, 14]
            head_feat = self.viz_encoder_2d(head_img)
            head_feat = self.in_proj(head_feat)            # [B*T*N, 256, 14, 14]
            feat_list.append(head_feat)

        if len(audio_input) and len(head_pos):
            # 2. Prepare audio imput: [B, T, 3, 210, 210] -> [B*T*N, 3, 210, 210]
            aud = audio_input
            B, T, C, H, W = aud.shape
            aud = aud.reshape(B * T, C, H, W)   # [B*T, 3, 210, 210]
            aud_input = torch.from_numpy(np.repeat(aud.cpu().numpy(), self.num_sub, axis=0)).cuda()    # [B*T*N, 3, 210, 210]

            # 3. Reshape head positions: [B, T, 4, 210, 210] -> [B*T*N, 1, 210, 210]
            msk = head_pos
            B, T, N, H, W = msk.shape
            msk_input = msk.reshape(B * T * N, H, W).unsqueeze(1) # [B*T*N, 1, 210, 210]

            # + pos input
            aud_msk_input = torch.cat((msk_input, aud_input), dim=1)    # [B*T*N, (1+3), 210, 210]

            # Extract audio feature: [B*T*N, 4, 210, 210] -> [B*T*N, 512, 14, 14]
            aud_feat = self.aud_pos_encoder_2d(aud_msk_input)  # [B*T*N, 512, 14, 14]
            aud_feat = self.in_proj(aud_feat)                  # channel 512 -> 256
            feat_list.append(aud_feat)

        if self.abla:
            BTN, C, H, W = feat_list[0].shape
            # [B, M, T, N, HW, C] -> [5, 1, 6, 4, 196, 256]
            av_feat = feat_list[0].reshape(BTN, C, H * W).permute(0, 2, 1).reshape(B, T, N, H * W, C).unsqueeze(1)
        else:
            feat_v, feat_a = feat_list[0], feat_list[1]
            # [BTN, C, H, W] -> [BTN, C, HW] -> [BTN, HW, C] -> [BTN, 16, C] -> [B, T, N, 16, C] -> [B, 1, T, N, 16, C]
            BTN, C, H, W = feat_v.shape
            if self.low_spa:
                # [B, M, T, N, 16, C] -> [5, 1, 6, 4, 16, 256]
                feat_v = self.spa_proj(feat_v.reshape(BTN, C, H * W).permute(0, 2, 1)).reshape(B, T, N, 16, C).unsqueeze(1)
                feat_a = self.spa_proj(feat_a.reshape(BTN, C, H * W).permute(0, 2, 1)).reshape(B, T, N, 16, C).unsqueeze(1)
            else:
                # [B, M, T, N, HW, C] -> [5, 1, 6, 4, 196, 256]
                feat_v = feat_v.reshape(BTN, C, H * W).permute(0, 2, 1).reshape(B, T, N, H * W, C).unsqueeze(1)
                feat_a = feat_a.reshape(BTN, C, H * W).permute(0, 2, 1).reshape(B, T, N, H * W, C).unsqueeze(1)
            av_feat = torch.cat([feat_v, feat_a], dim=1)    # [B, M, T, N, S, C]   M=2, S=16
        B, M, T, N, S, C = av_feat.shape

        """
        av_feat         torch.Size([5, 2, 6, 4, 196, 256])
        single_av_feat  torch.Size([20, 6, 512, 196]),  [BN, T, MC, S]
        fuse_feat       torch.Size([30, 6, 1024, 196]), [BN, T, MC, S]
        """
        temp_feat = av_feat.reshape(B, M, T, N, S, C)
        # [B, M, T, N, S, C] -> [B, N, S, T, M, C] -> [BNS, T, MC]
        temp_feat = av_feat.permute(0, 3, 4, 2, 1, 5).reshape(B * N * S, T, M * C)  # [BNS, T, MC]
        if self.attn_type != "None":
            temp_feat += self.temp_embed

        # [BNS, T, MC] -> [B, N, S, T, M, C] -> [B, T, S, N, M, C]-> [BTS, N, MC]
        sub_feat = temp_feat.reshape(B, N, S, T, M, C).permute(0, 3, 2, 1, 4, 5).reshape(B * T * S, N, M * C)
        if self.attn_type != "None":
            sub_feat += self.subj_embed

        # [BTS, N, MC] -> [B, T, S, N, M, C] -> [B, N, T, S, M, C] -> [BNT, S, MC]
        spat_feat = sub_feat.reshape(B, T, S, N, M, C).permute(0, 3, 1, 2, 4, 5).reshape(B * N * T, S, M * C)
        if self.attn_type != "None":
            spat_feat += self.spat_embed

        # [BNT, S, MC] -> [B, N, T, S, M, C] -> [B, M, T, N, S, C]
        av_feat = spat_feat.reshape(B, N, T, S, M, C).permute(0, 4, 2, 1, 3, 5)

        # go thru conversational attention module
        if self.attn_type == "None":    # DIRECT CONCAT
            # [B, M, T, N, S, C] -> [B, N, T, M, C, S] -> [BN, T, MC, S]
            single_av_feat = av_feat.permute(0, 3, 2, 1, 5, 4).reshape(B * N, T, M * C, S)
        elif self.attn_type in self.div_list:
            """ embeds: torch.Size([20, 1176, 512]), [BN, TS, MC] """
            # [B, M, T, N, S, C] -> [B, N, T, S, M, C] -> [BN, TS, MC]
            embeds = av_feat.permute(0, 3, 2, 4, 1, 5).reshape(B * N, T * S, M * C)
            embeds = self.conv_attn(embeds)
            # [BN, TS, MC] -> [BN, T, S, M, C] -> [BN, T, M, C, S] -> [BN, T, MC, S]
            single_av_feat = embeds.reshape(B * N, T, S, M, C).permute(0, 1, 3, 4, 2).reshape(B * N, T, M * C, S)
        elif self.attn_type == "TNS":
            """ embeds: torch.Size([5, 4704, 512]), [B, NTS, MC] """
            # [B, M, T, N, S, C] -> [B, N, T, S, M, C] -> [B, NTS, MC]
            embeds = av_feat.permute(0, 3, 2, 4, 1, 5).reshape(B, N * T * S, M * C)
            embeds = self.conv_attn(embeds)
            # [B, NTS, MC] -> [BN, T, S, M, C] -> [BN, T, M, C, S] -> [BN, T, MC, S]
            single_av_feat = embeds.reshape(B * N, T, S, M, C).permute(0, 1, 3, 4, 2).reshape(B * N, T, M * C, S)
        elif self.attn_type == "T":
            """ embeds  torch.Size([320, 6, 512]), [BNS, T, MC] """
            # [B, M, T, N, S, C] -> [B, N, S, T, M, C] -> [BNS, T, MC]
            embeds = av_feat.permute(0, 3, 4, 2, 1, 5).reshape(B * N * S, T, M * C)
            embeds = self.conv_attn(embeds)
            # [BNS, T, MC] -> [BN, S, T, MC] -> [BN, T, MC, S]
            single_av_feat = embeds.reshape(B * N, S, T, M * C).permute(0, 2, 3, 1)
        elif self.attn_type == "N":
            """ embeds  torch.Size([480, 4, 512]), [BTS, N, MC] """
            # [B, M, T, N, S, C] -> [B, T, S, N, M, C] -> [BTS, N, MC]
            embeds = av_feat.permute(0, 2, 4, 3, 1, 5).reshape(B * T * S, N, M * C)
            embeds = self.conv_attn(embeds)
            # [BTS, N, MC] -> [B, T, S, N, MC] -> [B, N, T, MC, S] -> [BN, T, MC, S]
            single_av_feat = embeds.reshape(B, T, S, N, M * C).permute(0, 3, 1, 4, 2).reshape(B * N, T, M * C, S)
        elif self.attn_type == "S":
            """ embeds  torch.Size([120, 196, 256]), [BNT, S, MC] """
            # [B, M, T, N, S, C] -> [B, N, T, S, M, C] -> [BNT, S, MC]
            embeds = av_feat.permute(0, 3, 2, 4, 1, 5).reshape(B * N * T, S, M * C)
            embeds = self.conv_attn(embeds)
            # [BNT, S, MC] -> [BN, T, S, MC] -> [BN, T, MC, S]
            single_av_feat = embeds.reshape(B * N, T, S, M * C).permute(0, 1, 3, 2)
        elif self.attn_type == "TN":
            """ embeds  torch.Size([80, 24, 512]), [BS, TN, MC] """
            # [B, M, T, N, S, C] -> [B, S, T, N, M, C] -> [BS, TN, MC]
            embeds = av_feat.permute(0, 4, 2, 3, 1, 5).reshape(B * S, T * N, M * C)
            embeds = self.conv_attn(embeds)
            # [BS, TN, MC] -> [B, S, T, N, MC] -> [B, N, T, MC, S] -> [BN, T, MC, S]
            single_av_feat = embeds.reshape(B, S, T, N, M * C).permute(0, 3, 2, 4, 1).reshape(B * N, T, M * C, S)
        elif self.attn_type == "NS":
            """ embeds  torch.Size([30, 784, 512]), [BT, NS, MC] """
            # [B, M, T, N, S, C] -> [B, T, N, S, M, C] -> [BT, NS, MC]
            embeds = av_feat.permute(0, 2, 3, 4, 1, 5).reshape(B * T, N * S, M * C)
            embeds = self.conv_attn(embeds)
            # [BT, NS, MC] -> [B, T, N, S, MC] -> [B, N, T, MC, S] -> [BN, T, MC, S]
            single_av_feat = embeds.reshape(B, T, N, S, M * C).permute(0, 2, 1, 4, 3).reshape(B * N, T, M * C, S)

        # pairwise feature fuse
        # [B*N, T, 512, 196] -> [B*M, T, 1024, 196]
        count = 0
        _, T, C, HW = single_av_feat.shape
        inter_dim = int(C * 2)
        fuse_feat = torch.zeros(B * 6, T, inter_dim, HW).cuda()

        for i in range(B):
            for j in self.idx_pair:
                id1, id2 = j[0] + i * 4, j[1] + i * 4
                feat1 = single_av_feat[id1]   # torch.Size([T, 512, 196])
                feat2 = single_av_feat[id2]   # torch.Size([T, 512, 196])
                fuse_feat[count] = torch.cat((feat1, feat2), dim=1)    # torch.Size([T, 1024, 196])
                count += 1

        if self.pair_attn != "None":
            BM, T, C, S = fuse_feat.shape
            # [BM, T, C, S] -> [BM, T, S, C] -> [BM, TS, C]
            embeds = fuse_feat.permute(0, 1, 3, 2).reshape(BM, T * S, C)
            embeds = self.pairwise_attn(embeds)
            # [BM, TS, C] -> [BM, T, S, C] -> [BM, T, C, S]
            fuse_feat = embeds.reshape(BM, T, S, C).permute(0, 1, 3, 2)

        # Classifier
        # [B*N, T, 512, 196] -> [B*N*T, 3]
        ego_spk_preds = self.cls_ego_spk(single_av_feat)
        sub_spk_preds = self.cls_sub_spk(single_av_feat)
        ego_lst_preds = self.cls_ego_lst(single_av_feat)
        sub_lst_preds = self.cls_sub_lst(single_av_feat)

        # [B*M, T, 1024, 196] -> [B*M*T, 3]
        id1_spk_preds = self.cls_id1_spk(fuse_feat)
        id2_spk_preds = self.cls_id2_spk(fuse_feat)
        id1_lst_preds = self.cls_id1_lst(fuse_feat)
        id2_lst_preds = self.cls_id2_lst(fuse_feat)

        ego_pred = [ego_spk_preds, sub_spk_preds, ego_lst_preds, sub_lst_preds]
        exo_pred = [id1_spk_preds, id2_spk_preds, id1_lst_preds, id2_lst_preds]

        return ego_pred, exo_pred
