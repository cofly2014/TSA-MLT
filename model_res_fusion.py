import torch
import torch.nn as nn
from utils import split_first_dim_linear
import math
from itertools import combinations

from torch.autograd import Variable

from utils import device

from backbone import MyResNet
from ttm_improved import TTM
import utils

NUM_SAMPLES = 1
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
import numpy as np

np.set_printoptions(threshold=np.inf)  # np.inf表示正无穷


class SimpleFCN(nn.Module):
    def __init__(self, input_channel):
        super(SimpleFCN, self).__init__()
        self.OneFCN = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
        )

    def forward(self, feature):
        out = self.OneFCN(feature)
        return out


# Self_Attn_Bot调用
class Bottleneck_Perceptron_3_layer_res(torch.nn.Module):
    '''
        3-layer Bottleneck MLP followed by a residual layer
    '''

    def __init__(self, in_dim):
        # in_dim 2048
        super(Bottleneck_Perceptron_3_layer_res, self).__init__()
        self.inp_fc = nn.Linear(in_dim, in_dim // 2)
        self.hid_fc = nn.Linear(in_dim // 2, in_dim // 2)
        self.out_fc = nn.Linear(in_dim // 2, in_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        output = self.relu(self.inp_fc(x))
        output = self.relu(self.hid_fc(output))
        output = self.out_fc(output)

        return output + x  # Residual output


class Self_Attn_Bot(nn.Module):
    """ Self attention Layer
        Attention-based frame enrichment
    """

    def __init__(self, in_dim, seq_len):
        super(Self_Attn_Bot, self).__init__()
        self.chanel_in = in_dim  # 2048

        # Using Linear projections for Key, Query and Value vectors
        self.key_proj = nn.Linear(in_dim, in_dim)
        self.query_proj = nn.Linear(in_dim, in_dim)
        self.value_conv = nn.Linear(in_dim, in_dim)

        self.softmax = nn.Softmax(dim=-1)  #
        self.gamma = nn.Parameter(torch.zeros(1))
        self.Bot_MLP = Bottleneck_Perceptron_3_layer_res(in_dim)
        max_len = int(seq_len * 1.5)
        self.pe = PositionalEncoding(in_dim, 0.1, max_len)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W )[B x 16 x 2048]
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width)
        """

        # Add a position embedding to the 16 patches
        x = self.pe(x)  # B x 16 x 2048

        m_batchsize, C, width = x.size()  # m = 200/160, C = 2048, width = 16

        # Save residual for later use
        residual = x  # B x 16 x 2048

        # Perform query projection
        proj_query = self.query_proj(x)  # B x 16 x 2048

        # Perform Key projection
        proj_key = self.key_proj(x).permute(0, 2, 1)  # B x 2048  x 16

        energy = torch.bmm(proj_query, proj_key)  # transpose check B x 16 x 16
        attention = self.softmax(energy)  # B x 16 x 16

        # Get the entire value in 2048 dimension
        proj_value = self.value_conv(x).permute(0, 2, 1)  # B x 2048 x 16

        # Element-wise multiplication of projected value and attention: shape is x B x C x N: 1 x 2048 x 8
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x 2048 x 16

        # Reshaping before passing through MLP
        out = out.permute(0, 2, 1)  # B x 16 x 2048

        # Passing via gamma attention
        out = self.gamma * out + residual  # B x 16 x 2048

        # Pass it via a 3-layer Bottleneck MLP with Residual Layer defined within MLP
        out = self.Bot_MLP(out)  # B x 16 x 2048

        return out


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000, pe_scale_factor=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_scale_factor = pe_scale_factor
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) * self.pe_scale_factor
        pe[:, 1::2] = torch.cos(position * div_term) * self.pe_scale_factor
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class TemporalCrossTransformer(nn.Module):
    def __init__(self, args, temporal_set_size=3):
        super(TemporalCrossTransformer, self).__init__()

        self.args = args
        self.temporal_set_size = temporal_set_size

        max_len = int(self.args.seq_len * 1.5)
        self.pe = PositionalEncoding(self.args.trans_linear_in_dim, self.args.trans_dropout, max_len=max_len)

        self.k_linear = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size,
                                  self.args.trans_linear_out_dim)  # .cuda()
        self.v_linear = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size,
                                  self.args.trans_linear_out_dim)  # .cuda()

        self.norm_k = nn.LayerNorm(self.args.trans_linear_out_dim)
        self.norm_v = nn.LayerNorm(self.args.trans_linear_out_dim)

        self.class_softmax = torch.nn.Softmax(dim=1)
        #########################################################################################
        # generate all tuples
        frame_idxs = [i for i in range(self.args.seq_len)]

        frame_combinations = combinations(frame_idxs, temporal_set_size)

        # 01 02 03... 12 13... 67
        self.tuples = [torch.tensor(comb).to(device) for comb in frame_combinations]
        self.tuples_len = len(self.tuples)
        ##################################################################
        #the array of the feature level
        self.temporal_set = self.args.temporal_set
        # 为每个分组
        self.k_linear_s = nn.ModuleList().to(device)
        self.v_linear_s = nn.ModuleList().to(device)

        self.level_number_set = self.args.level_number_set
        self.tuples_multi = []
        self.tuples_len_multi = torch.zeros(len(self.temporal_set)).to(device)
        self.total_combination_number = 0
        for i in range(len(self.temporal_set)):
            k_linear = nn.Linear(self.args.trans_linear_in_dim * self.temporal_set[i], self.args.trans_linear_out_dim)  # .cuda()
            self.k_linear_s.append(k_linear)
            v_linear = nn.Linear(self.args.trans_linear_in_dim * self.temporal_set[i],  self.args.trans_linear_out_dim)  # .cuda()
            self.v_linear_s.append(v_linear)
            #############################
            frame_combinations = combinations(frame_idxs, self.temporal_set[i])

            self.tuples_multi.append([torch.tensor(comb).to(device) for comb in frame_combinations])

            self.tuples_len_multi[i] = len(self.tuples_multi[i])

            self.total_combination_number = self.total_combination_number + self.level_number_set[i]

        self.total_combination_number = self.args.combination_number_used

        #################################################################################################

        self.fusion_linear_one = nn.Linear(10,5)  # .cuda()
        self.norm_fusion = nn.LayerNorm(10)
        self.batch_norm = nn.BatchNorm1d(5)
        self.fusionRelu = nn.ReLU()
        self.fusion_linear_two = nn.Conv1d(10, 5, 3, padding=1)
        #self.fusion_linear_two = nn.Linear(20,5)  # .cuda()
        #################################################################################################

    # context_features, context_labels, target_features
    def forward(self, support_set, support_labels, queries):
        n_queries = queries.shape[0]  # 25  8  2048
        n_support = support_set.shape[0]  # 25  8  2048

        # static pe
        support_set = self.pe(support_set)  # 5  8  512    类数*shot数   frame数量  表征向量长度
        queries = self.pe(queries)  # 25  8  512
        # torch.index_select(tensor,维度，选择的index）
        # construct new queries and support set made of tuples of images after pe
        # sss 一个list  每个元素为  5  2  512  从frame中选取2个frame的组合
        sss = [torch.index_select(support_set, -2, p) for p in self.tuples]
        # guofei
        # support_score = [torch.index_select(support_import_score, -1, p) for p in self.tuples]
        # s 合并 一个list中每个元素后两维=》  5   1024
        s = [torch.index_select(support_set, -2, p).reshape(n_support, -1) for p in self.tuples]
        # q 合并 一个list中每个元素后两维=》  25   1024
        q = [torch.index_select(queries, -2, p).reshape(n_queries, -1) for p in self.tuples]
        # guofei
        # query_score = [torch.index_select(query_import_score, -1, p) for p in self.tuples]

        #########################################
        support_set_r = support_set
        queries_r = queries
        support_num = support_set_r.shape[0]
        query_num = queries_r.shape[0]
        mh_support_set_ks = torch.empty((support_num, 0, self.args.trans_linear_out_dim), dtype=torch.float32).to(
            device)
        mh_queries_ks = torch.empty((query_num, 0, self.args.trans_linear_out_dim), dtype=torch.float32).to(device)
        mh_support_set_vs = torch.empty((support_num, 0, self.args.trans_linear_out_dim), dtype=torch.float32).to(
            device)
        mh_queries_vs = torch.empty((query_num, 0, self.args.trans_linear_out_dim), dtype=torch.float32).to(device)

        for i in range(len(self.temporal_set)):
            tuples = self.tuples_multi[i]
            s = [torch.index_select(support_set_r, -2, p).reshape(n_support, -1) for p in tuples]
            q = [torch.index_select(queries_r, -2, p).reshape(n_queries, -1) for p in tuples]
            support_set = torch.stack(s, dim=-2)  # 合并list   5   28   1024  其中28是所有的选取的2个frame的组合一共有28组
            queries = torch.stack(q, dim=-2)  # 合并list   25  28   1024  其中28是所有的选取的2个frame的组合一共有28组

            import random

            index_s = random.sample(range(0, len(s)), self.level_number_set[i])
            '''
            if i == 0:
                index_s = random.sample(range(0, len(s)), 8)
            if i == 1:
                index_s = random.sample(range(0, len(s)), 4)
            if i == 2:
                index_s = random.sample(range(0, len(s)), 3)
            if i == 3:
                index_s = random.sample(range(0, len(s)), 2)
            if i == 4:
                index_s = random.sample(range(0, len(s)), 1)
            '''
            indices = torch.tensor(index_s).to(device)
            support_set = torch.index_select(support_set, 1, indices)
            queries = torch.index_select(queries, 1, indices)

            # apply linear maps
            # support_set_ks  5  28  128 一个线性变换   queries_ks 25  28  128
            support_set_ks = self.k_linear_s[i](support_set)
            queries_ks = self.k_linear_s[i](queries)
            support_set_vs = self.v_linear_s[i](support_set)
            queries_vs = self.v_linear_s[i](queries)

            # apply norms where necessary
            '''
            mh_support_set_ks = torch.cat([mh_support_set_ks, self.norm_k(support_set_ks)], dim=1)
            mh_queries_ks = torch.cat([mh_queries_ks, self.norm_k(queries_ks)], dim=1)
            mh_support_set_vs = torch.cat([mh_support_set_vs, self.norm_k(support_set_vs)], dim=1)
            mh_queries_vs = torch.cat([mh_queries_vs, self.norm_k(queries_vs)], dim=1)
            '''
            mh_support_set_ks = torch.cat([mh_support_set_ks, support_set_ks], dim=1)
            mh_queries_ks = torch.cat([mh_queries_ks, queries_ks], dim=1)
            mh_support_set_vs = torch.cat([mh_support_set_vs, support_set_vs], dim=1)
            mh_queries_vs = torch.cat([mh_queries_vs, queries_vs], dim=1)

        mh_support_set_ks = self.norm_k(mh_support_set_ks)
        mh_queries_ks = self.norm_k(mh_queries_ks)
        mh_support_set_vs = self.norm_k(mh_support_set_vs)
        mh_queries_vs = self.norm_k(mh_queries_vs)

        unique_labels = torch.unique(support_labels)

        ############################################################
        ############################################################

        # init tensor to hold distances between every support tuple and every target tuple
        '''
        all_distances_tensor 第一维 query类数*每个类中样本数， 第二维为类别
                  类别1  类别2  类别3  类别4  类别5
         类别1样本1
         类别1样本2
         类别1样本3
         类别1样本4
         类别1样本5
         ...
         类别5样本5      
        '''
        all_distances_tensor = torch.zeros(n_queries, self.args.way).to(device)
        all_ot_distances_tensor = torch.zeros(n_queries, self.args.way).to(device)
        for label_idx, c in enumerate(unique_labels):
            # select keys and values for just this class
            '''
            mh_support_set_ks 5  28  128  第一维为类别，  class_k 为选取的第c类 1   28  128
            '''
            temp_t = self._extract_class_indices(support_labels, c)
            class_k = torch.index_select(mh_support_set_ks, 0, temp_t)
            '''
            mh_support_set_vs 5  28  128  第一维为类别，  class_v 为选取的第c类 1   28  128
            '''
            class_v = torch.index_select(mh_support_set_vs, 0, self._extract_class_indices(support_labels, c))

            ##########################################################################################################
            k_bs = class_k.shape[0]
            # mh_queries_ks.unsqueeze(1) => 25 ，1 ，28, 128  class_k.transpose(-2, -1)=> 1 128  28
            class_scores = torch.matmul(mh_queries_ks.unsqueeze(1), class_k.transpose(-2, -1)) / math.sqrt(
                self.args.trans_linear_out_dim)
            # reshape etc. to apply a softmax for each query tuple
            class_scores = class_scores.permute(0, 2, 1, 3)  # class_scores   25, 1 ,28,  28==> 25, 28, 1 28
            # self.tuples_len
            class_scores = class_scores.reshape(n_queries, self.total_combination_number, -1)  # 25  28  28
            class_scores = [self.class_softmax(class_scores[i]) for i in range(n_queries)]  # list长度为25 每个元素为[ 28,28]
            class_scores = torch.cat(class_scores)  # 700 28
            # self.tuples_len
            class_scores = class_scores.reshape(n_queries, self.total_combination_number, -1,
                                                self.total_combination_number)  # 25 28 1 28
            class_scores = class_scores.permute(0, 2, 1, 3)  # 25, 1 ,28,  28
            # print(class_scores[0][0].cpu().detach().numpy())
            # get query specific class prototype  #  class_v 为选取的第c类 1   28  128  ; class_scores 25, 1 ,28,  28
            # query_prototype  25  1 28  128
            query_prototype = torch.matmul(class_scores, class_v)
            query_prototype = torch.sum(query_prototype, dim=1)  # 25 ,28,  128

            # calculate distances from queries to query-specific class prototypes
            diff = mh_queries_vs - query_prototype
            norm_sq = torch.norm(diff, dim=[-2, -1]) ** 2
            # distance = torch.div(norm_sq, self.tuples_len)
            distance = torch.div(norm_sq, self.total_combination_number)

            # 做除法
            # multiply by -1 to get logits
            distance = distance * -1
            c_idx = c.long()
            all_distances_tensor[:, c_idx] = distance

            ##########'''The following code is for the ot distance'''###################

            ot_distances_tensor = torch.zeros(n_queries).to(device)

            for xx in range(mh_queries_vs.shape[0]):
                # 成本矩阵
                M = torch.cdist(mh_queries_vs[xx], query_prototype[xx], p=2.0,
                                compute_mode='use_mm_for_euclid_dist_if_necessary').to(device)
                n, m = M.shape
                r1 = Variable(torch.ones(n) / n).to(device)
                r2 = Variable(torch.ones(m) / m).to(device)
                P, ot_distance = utils.compute_optimal_transport_raw(M, r1, r2, lam=1, epsilon=1e-5)
                ot_distances_tensor[xx] = ot_distance
            # 做除法
            # multiply by -1 to get logits
            ot_distances_tensor = ot_distances_tensor * -1
            c_idx = c.long()
            all_ot_distances_tensor[:, c_idx] = ot_distances_tensor
            
        ########################################################################################
        fusion_distances_tensor = torch.cat([all_distances_tensor, all_ot_distances_tensor],dim=1)
        fusion_distances_tensor = self.fusion_linear_one(fusion_distances_tensor)
        fusion_distances_tensor = self.batch_norm(fusion_distances_tensor)

        return_dict = {'logits': {"l2": all_distances_tensor,
                                  "ot": all_ot_distances_tensor,
                                  "fusion": fusion_distances_tensor
                                  }
                       }

        return return_dict

    @staticmethod
    def _extract_class_indices(labels, which_class):
        """
        Helper method to extract the indices of elements which have the specified label.
        :param labels: (torch.tensor) Labels of the context set.
        :param which_class: Label for which indices are extracted.
        :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
        """
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector


class CNN_TSA_MLT(nn.Module):
    """
    Standard Resnet connected to a Temporal Cross Transformer.

    """

    def __init__(self, args):
        super(CNN_TSA_MLT, self).__init__()
        # 训练模式
        self.train()
        self.args = args
        self.args.num_patches = 16
        self.args.reduction_fac = 4

        ############################################backbone(ResNet50) and neck(getting 3 layers)######################
        self.resnet = MyResNet(self.args)
        self.adap_max = nn.AdaptiveMaxPool2d((4, 4))

        self.transformers = nn.ModuleList([TemporalCrossTransformer(args, 2)])

        # Temporal Cross Transformer for modelling temporal relations
        # self.transformers = nn.ModuleList([TemporalCrossTransformer(args, s) for s in args.temp_set])

        # 去掉标准resnet的最后一层
        # last_layer_idx = -1
        # self.resnet = nn.Sequential(*list(resnet.children())[:last_layer_idx])

        ######################################################################################################
        self.simpleFCN = SimpleFCN(self.args.trans_linear_in_dim)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #####################################################################################################
        # self.atm = TAM(in_channels = 2048, n_segment=8, kernel_size = 3, stride = 1, padding = 1).to(device)
        dim = (self.args.trans_linear_in_dim, self.args.trans_linear_in_dim)
        self.ttm_stage = TTM(self.args.seq_len, self.args.shot, dim).to(device)

        self.keynet_multi = nn.Conv1d(*dim, kernel_size=1, bias=False)
        self.querynet_multi = nn.Conv1d(*dim, kernel_size=1, bias=False)
        self.valuenet_multi = nn.Conv1d(dim[0], dim[0], kernel_size=1, bias=False)
        self.dim = (self.args.trans_linear_in_dim, self.args.trans_linear_in_dim)
        ###########################################################
        self.clsW = nn.Linear(self.args.trans_linear_in_dim, self.args.trans_linear_in_dim // 2).to(device)

        self.relu = torch.nn.ReLU()

    def forward(self, context_images, context_labels, target_images):
        ######################################################################################################
        '''
            context_features/target_features is of shape (num_images x 2048) [final Resnet FC layer] after squeezing
        '''
        '''
            context_images: 200 x 3 x 224 x 224, target_images = 160 x 3 x 224 x 224
        '''
        # 目前取最后一层的输出TODO：后面看是否有必要进行多层融合
        support_out = self.resnet(context_images)[-1]
        query_out = self.resnet(target_images)[-1]

        ########################using ttm begin########################################
        ####从resnet输出的表征，在时间维度上进行采样对齐，这是一个线性操作
        ###这里进行各种变换是为了适配ttm_stage

        support_shape_ttm = support_out.shape
        support_out_ttm = support_out.reshape(int(support_shape_ttm[0] / self.args.seq_len), self.args.seq_len,
                                              support_shape_ttm[1], support_shape_ttm[2], support_shape_ttm[3])
        support_out_ttm = support_out_ttm.permute(0, 2, 1, 3, 4)
        query_shape_ttm = query_out.shape
        query_out_ttm = query_out.reshape(int(query_shape_ttm[0] / self.args.seq_len), self.args.seq_len,
                                          query_shape_ttm[1],
                                          query_shape_ttm[2], query_shape_ttm[3])
        query_out_ttm = query_out_ttm.permute(0, 2, 1, 3, 4)
        task_adaptive = True
        if task_adaptive == True:
            # 计算时间区域对齐调值参数
            task_parameter = self.ttm_stage.create_task_adaptive_p(support_out_ttm, query_out_ttm)
            support_out_ttm = self.ttm_stage.align(support_out_ttm, task_parameter)
            query_out_ttm = self.ttm_stage.align(query_out_ttm, task_parameter)
        else:
            support_out_ttm = self.ttm_stage.align(support_out_ttm)
            query_out_ttm = self.ttm_stage.align(query_out_ttm)
        # 是否增肌对齐模块
        is_using_ttm = True
        if is_using_ttm == True:
            support_out = support_out_ttm
            query_out = query_out_ttm

        ##################using ttm end##############################################

        context_features = self.avgpool(support_out).squeeze()
        target_features = self.avgpool(query_out).squeeze()
        ############################################################################
        # 下面三行是TRX原生处理
        dim = int(context_features.shape[1])
        context_features = context_features.reshape(-1, self.args.seq_len, dim)
        target_features = target_features.reshape(-1, self.args.seq_len, dim)

        ########################################################################
        ########################################################################
        ''' compute the query_prototype_raw'''
        query_prototype_raw = torch.zeros(self.args.way, self.args.seq_len, context_features.shape[2]).to(
            device)  # 20 x
        unique_labels = torch.unique(context_labels)
        for label_idx, c in enumerate(unique_labels):
            # select keys and values for just this class
            '''
            mh_support_set_ks 5  28  128  第一维为类别，  class_k 为选取的第c类 1   28  128
            '''
            temp_t = self._extract_class_indices(context_labels, c)
            class_k = torch.index_select(context_features, 0, temp_t)
            query_prototype_raw[label_idx:, ] = torch.mean(class_k, dim=0)

        ############################################################################
        ############################################################################

        # 进行了resnet处理之后，用temporal Cross Transformer进行处理
        '''
        all_logits = [transformer(context_features, context_labels, target_features)['logits'] for transformer in self.transformers]
        all_logits = torch.stack(all_logits, dim=-1)
        sample_logits = all_logits
        sample_logits = torch.mean(sample_logits, dim=[-1])
        '''
        all_logits_two = [transformer(context_features, context_labels, target_features)['logits'] for transformer in self.transformers]
        # 确定l2 or ot
        l2_logits = [each["l2"] for each in all_logits_two]
        l2_logits = torch.stack(l2_logits, dim=-1)
        sample_l2_logits = l2_logits
        sample_l2_logits = torch.mean(sample_l2_logits, dim=[-1])
        # 增加OT
        ot_logits = [each["ot"] for each in all_logits_two]
        ot_logits = torch.stack(ot_logits, dim=-1)
        sample_ot_logits = ot_logits
        sample_ot_logits = torch.mean(sample_ot_logits, dim=[-1])

        fusion_logits = [each["fusion"] for each in all_logits_two]
        fusion_logits = torch.stack(fusion_logits, dim=-1)
        sample_fusion_logits = fusion_logits
        sample_fusion_logits = torch.mean(sample_fusion_logits, dim=[-1])

        # sample_logits  行为样本个数 列为每个样本每个类上的计算值 例如 [25 5]
        return_dict = {'l2_logits': split_first_dim_linear(sample_l2_logits, [NUM_SAMPLES, target_features.shape[0]]),
                       'ot_logits': split_first_dim_linear(sample_ot_logits, [NUM_SAMPLES, target_features.shape[0]]),
                       'fusion_logits': split_first_dim_linear(sample_fusion_logits, [NUM_SAMPLES, target_features.shape[0]]),
                       # 'sim_kl':split_first_dim_linear(sim_kl, [NUM_SAMPLES, target_features.shape[0]]),
                       # 'part_ot': split_first_dim_linear(part_ot_distances_tensor, [NUM_SAMPLES, part_ot_distances_tensor.shape[0]])
                       }
        return return_dict

    def compute_ot_distance(self, query_prototype_raw, target_features):
        part_ot_distances_tensor = torch.zeros(target_features.shape[0], self.args.way).to(device)
        part_ot_distances = torch.zeros(target_features.shape[0])
        for xx in range(query_prototype_raw.shape[0]):
            query_prototype_raw_single = query_prototype_raw[xx]
            for yy in range(target_features.shape[0]):
                target_feature = target_features[yy]

                x_abs = target_feature.norm(dim=1)
                x_aug_abs = query_prototype_raw_single.norm(dim=1)

                M = -torch.einsum('ik,jk->ij', target_feature, query_prototype_raw_single) / torch.einsum('i,j->ij', x_abs, x_aug_abs)

                # M = torch.cdist(target_feature, query_prototype_raw_single, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary').clone().to(device)
                M = Variable(M)
                n, m = M.shape
                r1 = Variable(torch.ones(n) / n).to(device)
                r2 = Variable(torch.ones(m) / m).to(device)
                P, ot_distance = utils.compute_optimal_transport_raw(M, r1, r2, lam=1, epsilon=1e-5)
                part_ot_distances[yy] = ot_distance
            part_ot_distances_tensor[:, xx] = part_ot_distances
        return part_ot_distances_tensor

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            self.resnet.to(device)
            self.resnet = torch.nn.DataParallel(self.resnet, device_ids=[i for i in self.args.gpus_use])

            self.transformers.to(device)
            # self.new_dist_loss_post_pat = [n.cuda(3) for n in self.new_dist_loss_post_pat]

            self.ttm_stage.to(device)
            # self.ttm_stage = torch.nn.DataParallel(self.ttm_stage, device_ids=[i for i in self.args.gpus_use])

    @staticmethod
    def _extract_class_indices(labels, which_class):
        """
        Helper method to extract the indices of elements which have the specified label.
        :param labels: (torch.tensor) Labels of the context set.
        :param which_class: Label for which indices are extracted.
        :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
        """
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector


from utils import loss, aggregate_accuracy

if __name__ == "__main__":
    class ArgsObject(object):
        def __init__(self):
            self.trans_linear_in_dim = 512
            self.trans_linear_out_dim = 128

            self.way = 5
            self.shot = 1
            self.query_per_class = 5

            self.trans_dropout = 0.1
            # 每个视频采样8帧
            self.seq_len = 8
            self.img_size = 84
            self.method = "resnet18"
            self.num_gpus = 1
            self.temp_set = [2, 3]

            self.loss = loss
            self.accuracy_fn = aggregate_accuracy


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = ArgsObject()
    torch.manual_seed(0)

    model = CNN_TSA_MLT(args).to(device)
    #
    support_imgs = torch.rand(args.way * args.shot * args.seq_len, 3, args.img_size, args.img_size).to(device)
    target_imgs = torch.rand(args.way * args.query_per_class * args.seq_len, 3, args.img_size, args.img_size).to(device)
    support_labels = torch.tensor([0, 1, 2, 3, 4]).to(device)
    query_labels = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]).to(device)
    print("Support images input shape: {}".format(support_imgs.shape))
    print("Target images input shape: {}".format(target_imgs.shape))
    print("Support labels input shape: {}".format(support_imgs.shape))

    out = model(support_imgs, support_labels, target_imgs)

    print("TRX returns the distances from each query to each class prototype.  Use these as logits.  Shape: {}".format(
        out['logits'].shape))
    target_logits = out['logits']
    target_lstm_logits = out['logits_lstm_similarity']

    task_logits_total = target_logits  # + 0.1* target_lstm_logits
    # task_accuracy = args.accuracy_fn(task_logits_total, support_labels)

    task_loss = args.loss(target_logits, query_labels, device) / args.tasks_per_batch
    task_lstm_loss = args.loss(target_lstm_logits, query_labels, device) / args.tasks_per_batch




