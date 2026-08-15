import os
import pandas as pd
import torch
import torch.nn as nn
import faiss
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from model import TwoTowerModel  # Ensure this matches model.py

# 1. Load Local Data
data_path = os.path.join("data", "u.data")
if not os.path.exists(data_path):
    raise FileNotFoundError("Put 'u.data' in the 'data' folder first!")

df = pd.read_csv(data_path, sep='\t', names=['user_id', 'movie_id', 'rating', 'ts'])
df = df[df['rating'] >= 4] # Keep only positive interactions

# 2. Encode IDs
user_enc, item_enc = LabelEncoder(), LabelEncoder()
df['user_id'] = user_enc.fit_transform(df['user_id'])
df['movie_id'] = item_enc.fit_transform(df['movie_id'])

n_users = df['user_id'].nunique()
n_items = df['movie_id'].nunique()

# 3. Model Initialization
# Match the emb_dim from your model.py (which is 64)
model = TwoTowerModel(n_users, n_items, emb_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

# 4. Training Loop
users = torch.tensor(df['user_id'].values, dtype=torch.long)
items = torch.tensor(df['movie_id'].values, dtype=torch.long)
labels = torch.ones(len(df))

print("Training started...")
model.train()
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(users, items)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch} Loss: {loss.item():.4f}")

# 5. SAVE ARTIFACTS
os.makedirs("data", exist_ok=True)
torch.save(model.state_dict(), "data/model.pth")

metadata = {
    'n_users': n_users,
    'n_items': n_items,
    'user_encoder': user_enc,
    'item_encoder': item_enc
}
with open("data/metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

# 6. FIXED FAISS INDEXING
print("Building FAISS index...")
model.eval()
with torch.no_grad():
    # Get all movie embeddings, convert to float32 (REQUIRED for FAISS)
    all_item_ids = torch.arange(n_items)
    item_embs = model.item_net(model.item_embedding(all_item_ids))
    item_embs = item_embs.detach().cpu().numpy().astype('float32')

# Ensure item_embs is 2D (N, Dim)
if len(item_embs.shape) == 1:
    item_embs = np.expand_dims(item_embs, axis=0)

dim = item_embs.shape[1]
print(f"Final Index Dimension: {dim}")

# Create index and add vectors
index = faiss.IndexFlatIP(dim)
index.add(item_embs) 

faiss.write_index(index, "data/retrieval_index.bin")
print("✅ All artifacts saved successfully to /data folder!")