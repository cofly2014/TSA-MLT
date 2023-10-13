import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import numpy as np
from utils import device
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.autograd import Variable

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

class SimpleFCN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleFCN, self).__init__()
        self.OneFCN = nn.Sequential(
            nn.Conv2d(input_dim, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            )
    def forward(self, feature):
        out = self.OneFCN(feature)
        return out

class TTM(nn.Module):
    '''
    1st stage: Temporal Transformation Module (TTM)
    =>Input:
    support: (N, C, T, H, W)   query: (N, C, T, H, W)
    <= Return:
    aligned_support: (N, C, T, H, W)   aligned_query: (N, C, T, H, W)
    '''
    def __init__(self,T,shot=1,dim=(64,64)):
        super().__init__()
        self.T=T
        self.dim=dim
        self.shot=shot
        #这里的location network只是在时间维度进行平移，缩放，所以只有两个输出参数
        self.locnet=torch.nn.Sequential(
            nn.Conv3d(dim[0],64,3,padding=1),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(2),
            nn.ReLU(),#5,4,4

            nn.Conv3d(64,128,3,padding=1),
            nn.BatchNorm3d(128),
            nn.MaxPool3d(2),
            nn.ReLU(),#3,2,2

            nn.AdaptiveMaxPool3d((1,1,1)),

            nn.Flatten(),#128
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Linear(32,2),
            nn.Tanh(),
        )#[B,2] 2:=(a,b)=>ax+b
        #为什么最后一个linear的w和b要初始化为0 ？？
        self.locnet[-2].weight.data.zero_()
        self.locnet[-2].bias.data.copy_(torch.tensor([2.,0]))
        # 任务自适应调制参数成成
        self.task_adaptive_parameer = torch.nn.Sequential(
            nn.Conv2d(dim[0], 32, kernel_size=1, padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU(),  # 5,4,4

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(),  # 3,2,2

            nn.AdaptiveMaxPool2d((1, 1)),

            nn.Flatten(),  # 128

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )


        self.locnet_re=torch.nn.Sequential(
            nn.Conv3d(dim[0],64,3,padding=1),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(2),
            nn.ReLU(),#5,4,4

            nn.Conv3d(64,128,3,padding=1),
            nn.BatchNorm3d(128),
            nn.MaxPool3d(2),
            nn.ReLU(),#3,2,2

            nn.AdaptiveMaxPool3d((1,1,1)),

            nn.Flatten(),#128

            nn.Linear(128,32),
            nn.ReLU(),
            nn.Linear(32,2),
            nn.Tanh(),

        )#[B,2] 2:=(a,b)=>ax+b
        #为什么最后一个linear的w和b要初始化为0 ？？
        self.locnet_re[-2].weight.data.zero_()
        self.locnet_re[-2].bias.data.copy_(torch.tensor([32.,0]))

        self.task_adaptive_parameer =torch.nn.Sequential(
            nn.Conv2d(dim[0],32,  kernel_size=1, padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU(),#5,4,4

            nn.Conv2d(32,64, kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(),#3,2,2

            nn.AdaptiveMaxPool2d((1,1)),

            nn.Flatten(),#128

            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,2),
        )

        self.atten = Self_Attn_Bot(2048, 2048)

    '''
    task adaptive的特征调制参数的生成
    input:
    feature： support和query的特征组合
    '''
    def create_task_adaptive_p(self, support_feature, query_feature):

        feature = torch.cat([support_feature, query_feature], dim = 0)
        n,C,T,H,W=feature.shape
        atten_feature_input = feature.view(n, C, T, H * W)
        atten_feature_input = atten_feature_input.permute(0, 2, 3, 1).contiguous()
        atten_feature_input = atten_feature_input.view(n*T, atten_feature_input.shape[2], atten_feature_input.shape[3])

        # task_level的调制
        feature_atten = self.atten(atten_feature_input)

        feature_atten = feature_atten.permute(0, 2, 1)
        feature_atten = feature_atten.reshape(feature_atten.shape[0], feature_atten.shape[1], H, W)
        task_parameter = self.task_adaptive_parameer(feature_atten).contiguous()
        #task_parameter = task_parameter.reshape(int(task_parameter.shape[0] / T), T, -1).permute(0, 2, 1)
        task_parameter = torch.mean(task_parameter, dim = 0)
        task_parameter = torch.softmax(task_parameter, dim=0)
        return task_parameter

    def align(self, feature, task_parameter, vis=False):
        #batch_size  channle  time  H W
        n,C,T,H,W=feature.shape
        #经过Localization网络
        theta = self.locnet(feature)
        theta_re = self.locnet_re(feature)
        device=theta.device
        #grid_t 第一维是样本，第二维代表沿着时间维度的网格
        grid_t=torch.linspace(start=-1.0,end=1.0,steps=self.T).to(device).unsqueeze(0).expand(n,-1)
        grid_t=grid_t.reshape(n,1,T,1) #source uniform coord  # (10 1 8 1)
        #theta 第一维是样本，第二维是每个样本的属性； 例如 [5,2]
        #注释 [grid_t,torch.ones_like(grid_t)]是一个数组，
        #torch.cat([grid_t, torch.ones_like(grid_t)], -1) 在最后一维进行合并  例如[5，1，8，2],和并得这一维度对应的都是1
        #注释 torch.einsum('bc,bhtc->bht',theta,torch.cat([grid_t,torch.ones_like(grid_t)],-1)) 例如是 [5，1，8] ... 10,2  10 1 8 2--> 10 1 8

        grid_t_re = torch.einsum('bc, bhtc->bht', theta_re, torch.cat([grid_t, torch.ones_like(grid_t)], -1)).unsqueeze(-1)  #
        grid_t_for=torch.einsum('bc, bhtc->bht', theta, torch.cat([grid_t, torch.ones_like(grid_t)], -1)).unsqueeze(-1)  # [5,1,8,1]
        grid_for=torch.cat([grid_t_for, torch.zeros_like(grid_t_for)-1.0], -1) # N*1*T*2 -> (t,-1.0)  # [5,1,8,2]
        grid_re = torch.cat([grid_t_re, torch.zeros_like(grid_t_re) - 1.0], -1)

        grid = task_parameter[0]*grid_re + task_parameter[1]*grid_for

        #use gird to wrap support   因为平移以及缩放是在T维上的，所以这里将每一帧的数据压缩到一个维度上是合理的
        feature=feature.transpose(-3,-4).reshape(n,T,-1)   # 5 512 8 7 7 --> 5 8 512 7 7 -->5 8 25088
        feature=feature.transpose(-1,-2).unsqueeze(-2) # N*C*1*T   5  25088 8--> 5  25088   1  8
        #grid 10 1, 8,2  在10个样本每个样本的  grid[][][] 坐标点上进行采样，采用的值对应于 grid的第2维和第三维度
        feature_aligned = F.grid_sample(feature, grid, align_corners=True) #N*C*1*T
        #                               N*C*T             N*T*C
        feature_aligned = feature_aligned.squeeze(-2).transpose(-1,-2).reshape(n,T,-1,H,W).transpose(-3,-4) # 5 512 8 7 7
        #                                  |S|*T*C*H*W
        if not vis:
            return feature_aligned
        else:
            return feature_aligned, theta.detach()

    def forward(self,support,query,vis=False):
        '''
        inputs must have shape of N*C*T*H*W
        return S*Q*T
        '''
        # support=support.mean([-1,-2])
        # query=query.mean([-1,-2])
        n,C,T,H,W=support.shape
        m=query.size(0)
        theta_support=self.locnet(support)
        theta_query=self.locnet(query)

        grid_t=torch.linspace(start=-1.0,end=1.0,steps=self.T).to(device).unsqueeze(0).expand(n,-1)
        grid_t=grid_t.reshape(n,1,T,1) #source uniform coord
        grid_t=torch.einsum('bc,bhtc->bht',theta_support,torch.cat([grid_t,torch.ones_like(grid_t)],-1)).unsqueeze(-1)
        grid=torch.cat([grid_t,torch.zeros_like(grid_t)-1.0],-1) # N*1*T*2 -> (t,-1.0)
        # grid=torch.min(torch.max(grid,-1*torch.ones_like(grid)),torch.ones_like(grid))
        grid_support=grid

        #use gird to wrap support
        support=support.transpose(-3,-4).reshape(n,T,-1)
        support=support.transpose(-1,-2).unsqueeze(-2) # N*C*1*T
        support_aligned=F.grid_sample(support,grid,align_corners=True) #N*C*1*T
        #                               N*C*T             N*T*C
        support_aligned=support_aligned.squeeze(-2).transpose(-1,-2)\
                                    .reshape(n,T,-1,H,W).transpose(-3,-4)
        #                                  |S|*T*C*H*W


        grid_t=torch.linspace(start=-1.0,end=1.0,steps=self.T).to(device).unsqueeze(0).expand(m,-1)
        grid_t=grid_t.reshape(m,1,T,1) #source uniform coord
        grid_t=torch.einsum('bc,bhtc->bht',theta_query,torch.cat([grid_t,torch.ones_like(grid_t)],-1)).unsqueeze(-1)

        grid=torch.cat([grid_t,torch.zeros_like(grid_t)-1.0],-1) # N*1*T*2 -> (t,-1.0)
        # grid=torch.min(torch.max(grid,-1*torch.ones_like(grid)),torch.ones_like(grid))
        grid_query=grid


        #use gird to wrap query
        query=query.transpose(-3,-4).reshape(m,T,-1)
        query=query.transpose(-1,-2).unsqueeze(-2) # N*C*1*T
        query_aligned=F.grid_sample(query,grid,align_corners=True) #N*C*1*T
        #                               N*C*T             N*T*C
        query_aligned=query_aligned.squeeze(-2).transpose(-1,-2)\
                                    .reshape(m,T,-1,H,W).transpose(-3,-4)
        #                                |Q|*T*C*H*W

        #support_aligned=support_aligned #.unsqueeze(1).expand(n,m,C,T,H,W)
        #query_aligned=query_aligned #.unsqueeze(0).expand(n,m,C,T,H,W)

        if vis:
            vis_dict={
                'grid_support':grid_support.clone().detach(),
                'grid_query':grid_query.clone().detach(),
                'theta_support':theta_support.clone().detach(),
                'theta_query':theta_query.clone().detach(),
            }
            return support_aligned, query_aligned, vis_dict
        else:
            return support_aligned, query_aligned


if __name__ == "__main__":
    class ArgsObject(object):
        def __init__(self):
            self.trans_linear_in_dim = 128
            self.trans_linear_out_dim = 128

            self.way = 5
            self.shot = 1
            self.query_per_class = 5

            self.trans_dropout = 0.1
            #每个视频采样8帧
            self.seq_len = 8
            self.img_size = 84
            self.method = "resnet18"
            self.num_gpus = 1


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = ArgsObject()


    torch.manual_seed(0)
    dim = (args.trans_linear_in_dim, args.trans_linear_in_dim)

    #

    support_imgs = torch.rand(args.way * args.shot , 128, args.seq_len,  args.img_size, args.img_size).to(device)
    target_imgs = torch.rand(args.way * args.query_per_class ,128,  args.seq_len, args.img_size, args.img_size).to(device)
    support_labels = torch.tensor([0, 1, 2, 3, 4]).to(device)
    query_labels = torch.tensor([0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4]).to(device)

    print("Support images input shape: {}".format(support_imgs.shape))
    print("Target images input shape: {}".format(target_imgs.shape))
    print("Support labels input shape: {}".format(support_imgs.shape))
    ttm_stage = TTM(args.seq_len, args.shot, dim).to(device)


    support_out = ttm_stage.align(support_imgs)
    query_out = ttm_stage.align(target_imgs)

    print("support_out= {}".format(support_out))
    print("query_out= {}".format(query_out))
