import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType

class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, x, norm_adj_matrix):
        return torch.sparse.mm(norm_adj_matrix, x)

class GATLayer(nn.Module):
    def __init__(self, input_dim, output_dim, alpha=0.2):
        super(GATLayer, self).__init__()
        self.W = nn.Parameter(torch.empty(size=(input_dim, output_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * output_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)
        attention = F.softmax(e, dim=1)
        attention = F.dropout(attention, 0.6, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime)

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.mm(Wh, self.a[:Wh.size(1)])
        Wh2 = torch.mm(Wh, self.a[Wh.size(1):])
        e = self.leakyrelu(Wh1 + Wh2.T)
        return e

class PGA_DRL(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(PGA_DRL, self).__init__(config, dataset)

        # Configurations
        self.latent_dim = config['embedding_size']
        self.n_layers = config['n_layers']
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        self.gamma = config['gamma']
        self.device = config['device']

        # User and Item counts
        self.n_users = dataset.user_num
        self.n_items = dataset.item_num

        # Embeddings for users and items
        self.user_embedding = nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.latent_dim)

        # Graph Convolution and Attention Layers
        self.gcn_layers = nn.ModuleList([GCNLayer() for _ in range(self.n_layers)])
        self.gat_layers = nn.ModuleList([GATLayer(self.latent_dim, self.latent_dim) for _ in range(self.n_layers)])

        # Layer weights for progressive fusion
        self.layer_weights = nn.Parameter(torch.ones(self.n_layers + 1))

        # Critic MLP
        self.critic_mlp = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.latent_dim*2 ),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_dim * 2, self.latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_dim, 1)
        )

        # Initialize parameters
        self.apply(xavier_uniform_initialization)
        self.norm_adj_matrix = self.get_norm_adj_matrix(dataset).to(self.device)

        # Loss function
        self.bpr_loss = BPRLoss()

        # Optimizers
        self.define_optimizers(config)

    def define_optimizers(self, config):
        params = list(self.user_embedding.parameters()) + \
                 list(self.item_embedding.parameters()) + \
                 list(self.gcn_layers.parameters()) + \
                 list(self.gat_layers.parameters()) + \
                 [self.layer_weights] + \
                 list(self.critic_mlp.parameters())
        self.optimizer = torch.optim.Adam(params, lr=config['learning_rate'], weight_decay=self.reg_weight)

    def get_norm_adj_matrix(self, dataset):
        user_np, item_np = dataset.inter_matrix(form='coo').nonzero()
        user_np = user_np.astype(np.int64)
        item_np = item_np.astype(np.int64)
        ratings = np.ones(len(user_np))
        n_nodes = self.n_users + self.n_items
        adj_mat = sp.coo_matrix(
            (ratings, (user_np, item_np + self.n_users)), shape=(n_nodes, n_nodes))
        adj_mat = adj_mat + adj_mat.T
        rowsum = np.array(adj_mat.sum(1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        norm_adj_mat = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt).tocoo()
        indices = torch.LongTensor([norm_adj_mat.row, norm_adj_mat.col])
        values = torch.FloatTensor(norm_adj_mat.data)
        shape = norm_adj_mat.shape
        norm_adj_matrix = torch.sparse.FloatTensor(indices, values, torch.Size(shape))
        return norm_adj_matrix

    def forward(self):
        # Concatenation of GCN and GAT outputs
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        gcn_embeddings = [ego_embeddings]
        gat_embeddings = [ego_embeddings]

        # GCN and GAT layer outputs
        for gcn_layer, gat_layer in zip(self.gcn_layers, self.gat_layers):
            gcn_embeddings.append(gcn_layer(gcn_embeddings[-1], self.norm_adj_matrix))
            gat_embeddings.append(gat_layer(gat_embeddings[-1], self.norm_adj_matrix))

        gcn_stack = torch.stack(gcn_embeddings, dim=1)
        gat_stack = torch.stack(gat_embeddings, dim=1)

        # Progressive fusion via concatenation
        layer_weights = F.softmax(self.layer_weights, dim=0)
        fused_embeddings = torch.sum((gcn_stack + gat_stack) * layer_weights.view(1, -1, 1), dim=1)

        user_fused_embeddings, item_fused_embeddings = torch.split(fused_embeddings, [self.n_users, self.n_items])
        return user_fused_embeddings, item_fused_embeddings

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_fused_embeddings, item_fused_embeddings = self.forward()
        u_embeddings = user_fused_embeddings[user]
        pos_embeddings = item_fused_embeddings[pos_item]
        neg_embeddings = item_fused_embeddings[neg_item]

        pos_scores = torch.sum(u_embeddings * pos_embeddings, dim=1)
        neg_scores = torch.sum(u_embeddings * neg_embeddings, dim=1)
        bpr_loss = self.bpr_loss(pos_scores, neg_scores)

        critic_value_pos = self.critic_mlp(torch.cat([u_embeddings, pos_embeddings], dim=1)).squeeze()
        critic_value_neg = self.critic_mlp(torch.cat([u_embeddings, neg_embeddings], dim=1)).squeeze()

        rewards = F.softplus(pos_scores - neg_scores).detach()
        critic_loss = F.mse_loss(critic_value_pos, rewards) + F.mse_loss(critic_value_neg, torch.zeros_like(critic_value_neg))

        loss = bpr_loss + critic_loss
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_fused_embeddings, item_fused_embeddings = self.forward()
        scores = torch.sum(user_fused_embeddings[user] * item_fused_embeddings[item], dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_fused_embeddings, item_fused_embeddings = self.forward()
        scores = torch.matmul(user_fused_embeddings[user], item_fused_embeddings.transpose(0, 1))
        return scores.view(-1)
