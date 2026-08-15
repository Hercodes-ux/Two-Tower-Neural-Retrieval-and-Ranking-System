import torch
import torch.nn as nn

class TwoTowerModel(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=64):
        super(TwoTowerModel, self).__init__()
        # User Tower
        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.user_net = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim)
        )
        # Item Tower
        self.item_embedding = nn.Embedding(n_items, emb_dim)
        self.item_net = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim)
        )

    def forward(self, user_ids, item_ids):
        user_vec = self.user_net(self.user_embedding(user_ids))
        item_vec = self.item_net(self.item_embedding(item_ids))
        return torch.sum(user_vec * item_vec, dim=1)