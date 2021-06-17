from helper import *
import torch.nn as nn


class DY_Conv2d(nn.Conv2d):

    def __init__(self, in_chan, out_chan, kernel_size=3,
            stride=1, padding=1, dilation=1, groups=1, bias=False,
            act=nn.ReLU(inplace=True), K=4,
            temperature=30, temp_anneal_steps=3000):
        super(DY_Conv2d, self).__init__(
            in_chan, out_chan * K, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)
        assert in_chan // 4 > 0
        self.K = K
        self.act = act
        self.se_conv1 = nn.Conv2d(in_chan, in_chan // 4, 1, 1, 0, bias=True)
        self.se_conv2 = nn.Conv2d(in_chan // 4, K, 1, 1, 0, bias=True)
        self.temperature = temperature
        self.temp_anneal_steps = temp_anneal_steps
        self.temp_interval = (temperature - 1) / temp_anneal_steps

    def get_atten(self, x):
        bs, _, h, w = x.size()
        atten = torch.mean(x, dim=(2, 3), keepdim=True)
        atten = self.se_conv1(atten)
        atten = self.act(atten)
        atten = self.se_conv2(atten)
        if self.training and self.temp_anneal_steps > 0:
            atten = atten / self.temperature
            self.temperature -= self.temp_interval
            self.temp_anneal_steps -= 1
        atten = atten.softmax(dim=1).view(bs, -1)
        return atten

    def forward(self, x):
        bs, _, h, w = x.size()
        atten = self.get_atten(x)

        out_chan, in_chan, k1, k2 = self.weight.size()
        W = self.weight.view(1, self.K, -1, in_chan, k1, k2)
        W = (W * atten.view(bs, self.K, 1, 1, 1, 1)).sum(dim=1)
        W = W.view(-1, in_chan, k1, k2)

        b = self.bias
        if not b is None:
            b = b.view(1, self.K, -1)
            b = (b * atten.view(bs, self.K, 1)).sum(dim=1).view(-1)

        x = x.view(1, -1, h, w)

        out = F.conv2d(x, W, b, self.stride, self.padding,
                self.dilation, self.groups * bs)
        out = out.view(bs, -1, out.size(2), out.size(3))
        return out


class InteractE(torch.nn.Module):
    """
	Proposed method in the paper. Refer Section 6 of the paper for mode details

	Parameters
	----------
	params:        	Hyperparameters of the model
	chequer_perm:   Reshaping to be used by the model

	Returns
	-------
	The InteractE model instance

	"""

    def __init__(self, params, chequer_perm, initial_entity_emb, initial_relation_emb, device,
                 dnn_hidden_units=(512,),
                 cin_layer_size=(512, 512,), seed=1024):
        super(InteractE, self).__init__()

        self.p = params
        self.ent_embed = torch.nn.Embedding(self.p.num_ent, self.p.embed_dim, padding_idx=None)

        self.rel_embed = torch.nn.Embedding(self.p.num_rel * 2, self.p.embed_dim, padding_idx=None)

        if self.p.pre_emb:
            self.ent_embed.weight.data.copy_(torch.FloatTensor(initial_entity_emb))
            self.rel_embed.weight.data.copy_(torch.FloatTensor(initial_relation_emb))
        else:
            xavier_normal_(self.rel_embed.weight)
            xavier_normal_(self.ent_embed.weight)

        self.bceloss = torch.nn.BCELoss()

        self.inp_drop3 = torch.nn.Dropout(self.p.inp_drop)
        self.p.perm = self.p.perm + 1
        #self.conv1 = torch.nn.Conv2d(self.p.perm, self.p.num_filt, kernel_size=self.p.ker_sz)
        self.conv1 = DY_Conv2d(self.p.perm, 64, self.p.ker_sz, 1, 1, bias=True)
        self.bn_conv1 = torch.nn.BatchNorm2d(64)
        self.feature_map_drop1 = torch.nn.Dropout2d(self.p.feat_drop)

        self.conv2 = DY_Conv2d(64, 128, self.p.ker_sz, 1, 1, bias=True)
        self.bn_conv2 = torch.nn.BatchNorm2d(128)
        self.feature_map_drop2 = torch.nn.Dropout2d(self.p.feat_drop)

        self.conv3 = DY_Conv2d(128, 256, self.p.ker_sz, 1, 1, bias=True)
        self.bn_conv3 = torch.nn.BatchNorm2d(256)
        self.feature_map_drop3 = torch.nn.Dropout2d(self.p.feat_drop)

        self.flat_sz = 20 * 20 * 256
        self.bn0 = torch.nn.BatchNorm2d(self.p.perm)

        self.fc = torch.nn.Linear(self.flat_sz, self.p.embed_dim)
        self.bn2 = torch.nn.BatchNorm1d(self.p.embed_dim)
        self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
        self.chequer_perm = chequer_perm




        self.merger = nn.Linear(3 * self.p.embed_dim, self.p.embed_dim)

        self.w1 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.w1_relu = nn.ReLU()

    def loss(self, pred, true_label=None, sub_samp=None):
        label_pos = true_label[0];
        label_neg = true_label[1:]
        loss = self.bceloss(pred, true_label)
        return loss

    def forward(self, sub, rel, neg_ents, strategy='one_to_x'):
        sub_emb = self.ent_embed(sub)
        rel_emb = self.rel_embed(rel)

        comb_emb = torch.cat([sub_emb, rel_emb], dim=1)
        chequer_perm = comb_emb[:, self.chequer_perm]
        stack_inp1 = chequer_perm.reshape((-1, self.p.perm-1, 2 * self.p.k_w, self.p.k_h))
        stack_inp2 = comb_emb.reshape((-1, 1, 2 * self.p.k_w, self.p.k_h))
        stack_inp = torch.cat([stack_inp1, stack_inp2], dim=1)
        cnn_input = self.bn0(stack_inp)
        cnn_input = self.inp_drop3(cnn_input)

        x = self.conv1(cnn_input)
        x = self.bn_conv1(x)
        x = F.relu(x)
        x = self.feature_map_drop1(x)

        x = self.conv2(x)
        x = self.bn_conv2(x)
        x = F.relu(x)
        x = self.feature_map_drop2(x)

        x = self.conv3(x)
        x = self.bn_conv3(x)
        x = F.relu(x)
        x = self.feature_map_drop3(x)

        x = x.view(-1, self.flat_sz)
        x = self.fc(x)

        x = self.bn2(x)
        x = F.relu(x)
        x = self.hidden_drop(x)


        if strategy == 'one_to_n':
            x = torch.mm(x, self.ent_embed.weight.transpose(1, 0))
        else:
            x = torch.mul(x.unsqueeze(1), self.ent_embed(neg_ents)).sum(dim=-1)

        pred = torch.sigmoid(x)

        return pred
