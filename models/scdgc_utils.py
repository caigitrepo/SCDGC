import math
from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import torch
from torch.nn.parameter import Parameter
from collections import OrderedDict



class Semantic(nn.Module):

    def __init__(self, num_classes, image_feature_dim, query_dim, intermediary_dim=1024):
        super(Semantic, self).__init__()
        self.num_classes = num_classes
        self.image_feature_dim = image_feature_dim
        self.query_dim = query_dim
        self.intermediary_dim = intermediary_dim
        self.fc_1 = nn.Linear(self.image_feature_dim, self.intermediary_dim, bias=False)
        self.fc_2 = nn.Linear(self.query_dim, self.intermediary_dim, bias=False)
        self.fc_3 = nn.Linear(self.intermediary_dim, self.intermediary_dim)
        self.fc_a = nn.Linear(self.intermediary_dim, 1)


    def forward(self, img_feature_map, query_features):
        batch_size = img_feature_map.size(0)
        convsize   = img_feature_map.size(3)

        f_wh_feature = img_feature_map.permute((0, 2, 3, 1)).contiguous().view(batch_size * convsize * convsize, -1)
        f_wh_feature = self.fc_1(f_wh_feature) # 18 x 1024
        f_qy_feature = self.fc_2(query_features).view(self.num_classes, 1, self.intermediary_dim) # 80 x 1024 - > 80 x 1 x 1024
        lb_feature   = self.fc_3(torch.tanh(f_qy_feature * f_wh_feature).view(-1, self.intermediary_dim)) # 80 x 18 x 1024 -> 1440 x 1024

        coefficient = self.fc_a(lb_feature) # 1440 x 1
        coefficient = coefficient.view(self.num_classes, batch_size, -1).transpose(0, 1) # 80 x 2 x 9 -> 2 x 80 x 9

        coefficient = torch.sigmoid(coefficient)  # 2 x 80 x 9
        # coefficient = torch.softmax(coefficient, dim = -1)
        img_feature_map = img_feature_map.permute(0, 2, 3, 1).view(batch_size, convsize * convsize, -1) # 2 x 3 x 3 x 2048 -> 2 x 9 x 2048

        # 中间输出
        self.semantic_map = coefficient.permute(0, 2, 1).view(batch_size, convsize,convsize, self.num_classes, 1).detach().cpu()

        graph_net_input = torch.bmm(coefficient, img_feature_map)   # 2 x 80 x 2048

        return graph_net_input.permute(0, 2, 1) # 2 x 2048 x 80

class Relation(nn.Module):

    def __init__(self, num_classes, relation_dim, query_dim, intermediary_dim=1024):
        super(Relation, self).__init__()
        self.num_classes = num_classes
        self.relation_dim = relation_dim
        self.query_dim = query_dim
        self.intermediary_dim = intermediary_dim

        self.relation_convert = nn.Linear(relation_dim, intermediary_dim, bias = False)
        self.query2query = nn.Linear(query_dim * 2, intermediary_dim)
        self.query2ruler = nn.Linear(query_dim * 2, intermediary_dim)


    def forward(self, relation_map, query_features):
        inter_dim = self.intermediary_dim
        nb_class = self.num_classes
        batch_size = relation_map.size()[0]
        conv_size = relation_map.size()[3]
        query_dim = self.query_dim

        joint_repr_a = query_features.view(nb_class, 1, query_dim).repeat(1, nb_class, 1)
        joint_repr_b = query_features.view(1, nb_class, query_dim).repeat(nb_class, 1, 1)
        joint_repr = torch.cat([joint_repr_a, joint_repr_b], dim = -1).view(nb_class * nb_class, -1)
        query = self.query2query(joint_repr) # nb_class * nb_class, inter_dim
        ruler = self.query2ruler(joint_repr) # nb_class * nb_class, inter_dim

        _relation_map = self.relation_convert(
            relation_map.permute(0, 2, 3, 1).contiguous().view(batch_size * conv_size * conv_size, -1)
        ) # batch * conv * conv, inter_dim
        coefficient = torch.mm(_relation_map, query.permute(1, 0)) / inter_dim**0.5 # batch * conv * conv, nb_class * nb_class
        # self.correlation_map = coefficient.view(-1, nb_class, conv_size, conv_size).detach().cpu()

        coefficient = coefficient.view(batch_size, conv_size * conv_size, nb_class * nb_class) # batch, conv * conv, nb_class * nb_class
        # coefficient = torch.softmax(coefficient, dim = 1)
        coefficient = torch.sigmoid(coefficient)
        _relation_map = _relation_map.view(batch_size, conv_size * conv_size, inter_dim).permute(0, 2, 1) # batch, inter_dim, conv * conv
        query_relation = torch.bmm(_relation_map, coefficient).permute(0, 2, 1) # batch, nb_class * nb_class, inter_dim
        ruler = ruler.view(1, nb_class * nb_class, inter_dim).repeat(batch_size, 1, 1) # batch, nb_class * nb_class, inter_dim
        # ruler = torch.tanh(ruler) 
        relation = (query_relation * ruler).sum(-1) / inter_dim**0.5 # batch, nb_class * nb_class
        adjacent = relation.view(batch_size, nb_class, nb_class)
        
        self.relation = adjacent
        # return torch.sigmoid(adjacent)
        return adjacent # b x 80 x 80


class SConv(nn.Module): 
    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(dim, dim, 1), 
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.2),
            nn.Conv1d(dim, dim, 1),
            nn.BatchNorm1d(dim),
        )
    
    def forward(self, x):
        return self.body(x)


class GConv(nn.Module): 
    def __init__(self, dim):
        super().__init__()
        self.body_head = nn.Sequential(
            nn.Conv1d(dim, dim, 1), 
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.2),
        )
        self.mid = nn.Sequential(
            nn.BatchNorm1d(dim), 
            nn.LeakyReLU(0.2),
        )
        self.body_tail = nn.Sequential(
            nn.Conv1d(dim, dim, 1),
            nn.BatchNorm1d(dim),
        )
        
    def forward(self, x, adjacent):
        x = self.body_head(x)
        x = torch.bmm(x, adjacent)
        x = self.mid(x)
        x = self.body_tail(x)
        return x



class Reason(nn.Module):
    def __init__(self, input_dim, time_step = 1):
        super().__init__()
        self.input_dim = input_dim
        self.time_step = time_step
        self.sconvs = nn.ModuleList([SConv(input_dim) for _ in range(time_step)])
        self.gconvs = nn.ModuleList([GConv(input_dim) for _ in range(time_step)])

    def forward(self, input, adjacent):
        # input: b x 2048 x 80
        adjacent = adjacent.permute(0, 2, 1)
        # adjacent = adjacent + torch.eye(adjacent.shape[-1]).cuda()
        # adjacent = (adjacent >= 0.5).float()
        # print(adjacent[0])
        # b = adjacent.shape[0]
        # cls = adjacent.shape[-1]
        # adjacent = torch.eye(cls).view(1, cls, cls).repeat(b, 1, 1).cuda()

        # input = input.view(batch_size, nb_class, -1) # batch, nb_class, input_dim
        h = input
        for t in range(self.time_step):
            _neighbor = self.gconvs[t](h, adjacent)
            _h = self.sconvs[t](h)
            h = F.elu(h + _neighbor + _h)
        return h



class Element_Wise_Layer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Element_Wise_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        for i in range(self.in_features):
            self.weight[i].data.uniform_(-stdv, stdv)
        if self.bias is not None:
            for i in range(self.in_features):
                self.bias[i].data.uniform_(-stdv, stdv)

    def forward(self, input):
        x = input * self.weight
        x = torch.sum(x, 2)
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)
