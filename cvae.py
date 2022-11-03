#CVAE for a single label group

import torch
import torch.nn as nn
from torch.nn import init
from collections import OrderedDict
import torch.nn.functional as F
from utils import reparameterize, group_wise_reparameterize, accumulate_group_evidence
from model.blocks import LinearNorm
from torch.autograd import Variable

class CVAEnet(nn.Module):
    def __init__(self, x_dim_acc, x_dim_spk, y_dim_acc=10, y_dim_spk=10, z_dim_acc=256, z_dim_spk=256, n_classes_acc=6, n_classes_spk=24):
        super(CVAEnet, self).__init__()

        #ACC
        self.x_dim_acc=x_dim_acc
        self.y_dim_acc=y_dim_acc
        self.z_dim_acc=z_dim_acc
        self.n_classes_acc=n_classes_acc


        self.embedding_layer_acc=nn.Embedding(self.n_classes_acc,self.y_dim_acc)

        self.linear_model_acc = nn.Sequential(OrderedDict([
            ('linear_1', nn.Linear(in_features=self.x_dim_acc+self.y_dim_acc, out_features=256, bias=True)),
            ('tan_h_1', nn.Tanh())
            ]))

        self.mu_layer_acc = LinearNorm(
                256, 
                z_dim_acc)
        self.logvar_layer_acc = LinearNorm(
                256,
                z_dim_acc)

        #SPK
        self.x_dim_spk=x_dim_spk
        self.y_dim_spk=y_dim_spk
        self.z_dim_spk=z_dim_spk
        self.n_classes_spk=n_classes_spk


        self.embedding_layer_spk=nn.Embedding(self.n_classes_spk,self.y_dim_spk)

        self.linear_model_spk = nn.Sequential(OrderedDict([
            ('linear_1', nn.Linear(in_features=self.x_dim_spk+self.y_dim_spk, out_features=256, bias=True)),
            ('tan_h_1', nn.Tanh())
            ]))

        self.mu_layer_spk = LinearNorm(
                256, 
                z_dim_spk)
        self.logvar_layer_spk = LinearNorm(
                256,
                z_dim_spk)


        # weight initialization
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, label_acc, label_spk):
        x = x.view(x.size(0), -1)

        #ACC
        y_acc = self.embedding_layer_acc(label_acc)

        x_acc = self.linear_model_acc(torch.cat([x,y_acc],axis=1))
        
        x_mu_acc = self.mu_layer_acc(x_acc)
        x_logvar_acc = self.logvar_layer_acc(x_acc)
                
        z_acc = reparameterize(training=True, mu=x_mu_acc, logvar=x_logvar_acc)


        #SPK
        y_spk = self.embedding_layer_spk(label_spk)

        x_spk = self.linear_model_spk(torch.cat([x,y_spk],axis=1))
        
        x_mu_spk = self.mu_layer_spk(x_spk)
        x_logvar_spk = self.logvar_layer_spk(x_spk)
                
        z_spk = reparameterize(training=True, mu=x_mu_spk, logvar=x_logvar_spk)
                
        # cat_prob = self.categorical_layer(class_latent_embeddings)

        return (z_acc, y_acc, z_spk, y_spk, (x_mu_acc, x_logvar_acc, x_mu_spk, x_logvar_spk))

    def inference(self, x, label_acc, label_spk):
        x = x.view(x.size(0), -1)

        #ACC
        y_acc = self.embedding_layer_acc(label_acc)

        x_acc = self.linear_model_acc(torch.cat([x,y_acc],axis=1))
        
        x_mu_acc = self.mu_layer_acc(x_acc)
        x_logvar_acc = self.logvar_layer_acc(x_acc)
                
        z_acc = x_mu_acc
        # z_acc = reparameterize(training=False, mu=x_mu_acc, logvar=x_logvar_acc)


        #SPK
        y_spk = self.embedding_layer_spk(label_spk)

        x_spk = self.linear_model_spk(torch.cat([x,y_spk],axis=1))
        
        x_mu_spk = self.mu_layer_spk(x_spk)
        x_logvar_spk = self.logvar_layer_spk(x_spk)
                

        z_spk = x_mu_spk

        # z_spk = reparameterize(training=False, mu=x_mu_spk, logvar=x_logvar_spk)
                
        # cat_prob = self.categorical_layer(class_latent_embeddings)

        return (z_acc, y_acc, z_spk, y_spk, (x_mu_acc, x_logvar_acc, x_mu_spk, x_logvar_spk))





# model_config["reference_encoder"]["ref_enc_gru_size"], model_config["encoder"]["encoder_embedding_dim"], model_config["accent_encoder"]["n_accent_classes"]


