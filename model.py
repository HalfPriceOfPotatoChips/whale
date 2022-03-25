from torch import nn
import torch
import math
import torch.nn.functional as F
import timm
from config import config

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0,
                 m=0.50, easy_margin=False, ls_eps=0.0):
        '''
        in_features: dimension of the input
        out_features: dimension of the last layer (in our case the classification)
        s: norm of input feature
        m: margin
        ls_eps: label smoothing'''

        super(ArcMarginProduct, self).__init__()
        self.in_features, self.out_features = in_features, out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        # Fills the input `Tensor` with values according to the method described in
        # `Understanding the difficulty of training deep feedforward neural networks`
        # Glorot, X. & Bengio, Y. (2010)
        # using a uniform distribution.
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------
        one_hot = torch.zeros(cosine.size()).to(config['device'])

        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Applies 2D average-pooling operation in kH * kW regions by step size
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


class HappyWhaleModel(nn.Module):
    def __init__(self, modelName, numClasses, embeddingSize):

        super(HappyWhaleModel, self).__init__()
        # Retrieve pretrained weights
        self.backbone = timm.create_model(modelName, pretrained=True)
        # Save the number features from the backbone
        ### different models have different numbers e.g. EffnetB3 has 1536
        backbone_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()  # ?????
        self.backbone.global_pool = nn.Identity()  # ?????

        # for p in self.parameters():
        #     p.requires_grad = False

        self.gem = GeM()
        # Embedding layer (what we actually need)
        self.embedding = nn.Sequential(
                                       nn.Linear(backbone_features, embeddingSize),
                                       nn.BatchNorm1d(embeddingSize),
                                       nn.ReLU())
                                       #nn.Dropout(p=0.2))
        # self.embadding = nn.Linear(backbone_features, numClasses)
        self.arcface = ArcMarginProduct(in_features=embeddingSize,
                                        out_features=numClasses,
                                        s=30.0, m=0.50, easy_margin=False, ls_eps=0.0)

    def forward(self, image, target=None, prints=False):
        '''If there is a target it means that the model is training on the dataset.
        If there is no target, that means the model is predicting on the test dataset.
        In this case we would skip the ArcFace layer and return only the image embeddings.
        '''

        features = self.backbone(image)
        # flatten transforms from e.g.: [3, 1536, 1, 1] to [3, 1536]
        gem_pool = self.gem(features).flatten(1)
        embedding = self.embedding(gem_pool)
        if target != None:
            out = self.arcface(embedding, target)

        if prints:
            print("0. IN:", "image shape:", image.shape, "target:", target)
            print("1. Backbone Output:", features.shape)
            print("2. GeM Pool Output:", gem_pool.shape)
            print("3. Embedding Output:", embedding.shape)
            if target != None:
                print("4. ArcFace Output:", out.shape)

        if target != None:
            return out, embedding
        else:
            return embedding

# model = HappyWhaleModel(config['model_name'], config['num_class'], config['embedding_size'])
# print(len(list(model.parameters())))
# print(model)