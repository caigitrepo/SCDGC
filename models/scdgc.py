import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from models.scdgc_utils import Semantic, Relation, Reason, Element_Wise_Layer
from models.backbone import build_backbone
from pathlib import Path
from hydra.utils import get_original_cwd

class SCDGC(nn.Module):
    def __init__(self, args):
        super(SCDGC, self).__init__()
        self.args = args
        self.image_feature_dim = 2048
        self.intermediary_dim = 1024
        self.output_dim = 2048
        self.time_step = args.time_step
        if args.word_embedding:
            self.query_dim = 300
            self.category_query = nn.Parameter( self._load_features(), requires_grad = False )
        else:
            self.query_dim = args.num_classes
            self.category_query = nn.Parameter( torch.eye(self.args.num_classes), requires_grad = False )

        self.conv_semantic   = nn.Conv2d(self.image_feature_dim, self.image_feature_dim, 1, padding = 0)
        self.conv_relation   = nn.Conv2d(self.image_feature_dim, self.image_feature_dim, 3, padding = 0) # 3 or 1
        self.semantic_net    = Semantic(args.num_classes, self.image_feature_dim, self.query_dim)
        self.relation_net    = Relation(args.num_classes, self.image_feature_dim, self.query_dim, args.correlation_dim)
        self.reason_net      = Reason(self.image_feature_dim, self.time_step)
        self.conv_output     = nn.Conv1d(2 * self.image_feature_dim, self.output_dim, 1)
        #self.extra_cls       = nn.Linear(self.output_dim, 1)
        self.classifiers     = Element_Wise_Layer(args.num_classes, self.output_dim)

        self.avgpool2d       = nn.AvgPool2d(2, stride=2)
        self.maxpool2d       = nn.AdaptiveMaxPool2d(1)
        
        self._load_backbone_model()


    def forward(self, x):
        batch_size = x.size()[0]
        init_feature_map = self.backbone(x)
        # img_feature_map = self.avgpool2d(init_feature_map)
        img_feature_map = init_feature_map
        global_feature  = self.maxpool2d(init_feature_map)

        semantic_map = self.conv_semantic(img_feature_map)
        relation_map = self.conv_relation(img_feature_map)

        semantic_net_output = self.semantic_net(semantic_map, self.category_query)

        relation_net_output = self.relation_net(relation_map, self.category_query)
        reason_net_output   = self.reason_net(semantic_net_output, torch.sigmoid(relation_net_output))

        self.relation_with_logits = relation_net_output

        feat = torch.cat((
            reason_net_output, global_feature.squeeze(3).repeat(1, 1, self.args.num_classes)
        ), 1)


        feat  = self.conv_output(feat)
        feat  = torch.tanh(feat)
        feat  = feat.contiguous().permute(0, 2, 1)
        cls   = self.classifiers(feat).view(batch_size, self.args.num_classes)
        # extra = self.extra_cls(feat).squeeze(-1)

        output = cls #+ extra
        return output, relation_net_output

    def _load_backbone_model(self):
        self.backbone = build_backbone()
        alias = self.backbone.backbone.body
        for param in alias.parameters():
            param.requires_grad = False
        for param in alias.layer4.parameters():
            param.requires_grad = True

    def _load_features(self):
        word_file = Path(get_original_cwd(), "data", self.args.data + "_embeddings.npy").as_posix()
        return Variable(torch.from_numpy(np.load(word_file).astype(np.float32))).cuda()

    def train_mode(self):
        alias = self.backbone.backbone.body
        alias.eval()
        alias.layer4.train()
    



