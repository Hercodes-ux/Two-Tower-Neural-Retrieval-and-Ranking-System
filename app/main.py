import os
import torch
import pickle
import faiss
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# Import your professional modules
from src.model import TwoTowerModel
from src.ranker import FineRanker

app = FastAPI(title="NeuralStream | Industrial Personalization Engine")
ranker = FineRanker()

# --- ARTIFACT LOADING (ROBUST PATHING) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# 1. Load Metadata & Encoders
metadata_path = os.path.join(DATA_DIR, "metadata.pkl")
with open(metadata_path, "rb") as f:
    metadata = pickle.load(f)

# 2. Initialize Model (Using 64-dim to match your successful training)
model = TwoTowerModel(metadata['n_users'], metadata['n_items'], emb_dim=64)
model.load_state_dict(torch.load(os.path.join(DATA_DIR, "model.pth")))
model.eval()

# 3. Load FAISS Index
index = faiss.read_index(os.path.join(DATA_DIR, "retrieval_index.bin"))

# --- PROFESSIONAL UI (DARK MODE & REAL DATA) ---
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NeuralStream Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap" rel="stylesheet">
        <style>
            body { font-family: 'Plus Jakarta Sans', sans-serif; background-color: #050505; color: #ffffff; }
            .glass { background: rgba(20, 20, 20, 0.6); backdrop-filter: blur(20px); border: 1px solid rgba(255,255,255,0.05); }
            .netflix-red { color: #E50914; }
            .btn-red { background-color: #E50914; transition: all 0.2s ease; }
            .btn-red:hover { background-color: #f40a15; transform: translateY(-2px); box-shadow: 0 10px 20px -10px rgba(229, 9, 20, 0.5); }
            .card-hover { transition: all 0.3s ease; border: 1px solid transparent; }
            .card-hover:hover { border-color: #E50914; background: rgba(229, 9, 20, 0.05); }
        </style>
    </head>
    <body class="min-h-screen bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-gray-900 via-black to-black">
        <div class="max-w-6xl mx-auto px-6 py-12">
            
            <header class="flex flex-col md:flex-row justify-between items-start md:items-center mb-16 gap-6">
                <div>
                    <h1 class="text-5xl font-extrabold tracking-tighter italic"><span class="netflix-red">NEURAL</span>STREAM</h1>
                    <p class="text-gray-400 mt-2 font-light text-lg">Two-Tower Neural Retrieval & Industrial Ranking System</p>
                </div>
                <div class="flex items-center gap-3 glass px-4 py-2 rounded-full border border-green-500/30">
                    <div class="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                    <span class="text-green-500 text-xs font-bold uppercase tracking-widest">Live Inference Active</span>
                </div>
            </header>

            <main class="grid grid-cols-1 lg:grid-cols-12 gap-12">
                <!-- Sidebar -->
                <div class="lg:col-span-4 space-y-6">
                    <div class="glass p-8 rounded-3xl">
                        <h2 class="text-xl font-bold mb-6">System Controls</h2>
                        <div class="space-y-5">
                            <div>
                                <label class="text-xs font-bold text-gray-500 uppercase mb-2 block">Subscriber ID</label>
                                <input id="userId" type="number" placeholder="Enter ID (e.g. 160)" 
                                    class="w-full bg-black/50 border border-gray-800 rounded-xl p-4 text-white focus:ring-2 focus:ring-red-600 focus:outline-none">
                            </div>
                            <div>
                                <label class="text-xs font-bold text-gray-500 uppercase mb-2 block">Candidate K-Count</label>
                                <input id="kCount" type="number" value="50" class="w-full bg-black/50 border border-gray-800 rounded-xl p-4 text-white focus:outline-none">
                            </div>
                            <button onclick="fetchRecommendations()" class="w-full btn-red text-white font-bold py-4 rounded-xl text-lg">Generate Picks</button>
                        </div>
                    </div>
                </div>

                <!-- Results -->
                <div class="lg:col-span-8">
                    <div id="loader" class="hidden flex flex-col items-center justify-center py-20 space-y-4">
                        <div class="w-12 h-12 border-4 border-red-600 border-t-transparent rounded-full animate-spin"></div>
                        <p class="text-gray-400">Querying Latent Vector Space...</p>
                    </div>
                    <div id="resultsGrid" class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div class="col-span-full border-2 border-dashed border-gray-800 rounded-3xl p-20 text-center">
                            <p class="text-gray-600 italic">Results are generated via real-time dot-product similarity</p>
                        </div>
                    </div>
                </div>
            </main>
        </div>

        <script>
            async function fetchRecommendations() {
                const uId = document.getElementById('userId').value;
                const k = document.getElementById('kCount').value;
                const grid = document.getElementById('resultsGrid');
                const loader = document.getElementById('loader');

                if(!uId) return;
                loader.classList.remove('hidden');
                grid.classList.add('hidden');
                grid.innerHTML = '';

                try {
                    const response = await fetch(`/recommend/${uId}?k=${k}`);
                    const data = await response.json();
                    
                    loader.classList.add('hidden');
                    grid.classList.remove('hidden');

                    if(data.error) { alert(data.error); return; }

                    // REAL DATA NORMALIZATION: Scale confidence based on actual model scores
                    const allScores = data.results.map(r => r.score);
                    const maxS = Math.max(...allScores);
                    const minS = Math.min(...allScores);

                    data.results.slice(0, 10).forEach((item, index) => {
                        // Math: Normalize score into 0-100% relative to the candidate set
                        const confidence = maxS === minS ? 100 : (((item.score - minS) / (maxS - minS)) * 100).toFixed(2);
                        
                        const card = `
                            <div class="glass p-6 rounded-2xl card-hover relative overflow-hidden group">
                                <div class="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-100 transition-opacity">
                                    <span class="text-4xl font-black text-white italic">#${index + 1}</span>
                                </div>
                                <div class="relative z-10">
                                    <h3 class="text-gray-500 font-bold text-[10px] uppercase">Neural Match</h3>
                                    <p class="text-2xl font-bold mt-1 mb-4 text-white font-mono">ID: ${item.id}</p>
                                    <div class="flex justify-between items-end">
                                        <div>
                                            <p class="text-[9px] text-gray-500 uppercase">Dot Product</p>
                                            <p class="text-green-400 font-mono font-bold text-sm">${item.score.toFixed(4)}</p>
                                        </div>
                                        <div class="text-right">
                                            <p class="text-[9px] text-gray-500 uppercase tracking-widest">Confidence</p>
                                            <p class="text-white font-bold text-sm">${confidence}%</p>
                                        </div>
                                    </div>
                                    <div class="w-full h-1 bg-gray-800 rounded-full mt-3 overflow-hidden">
                                        <div class="h-full bg-red-600" style="width: ${confidence}%"></div>
                                    </div>
                                </div>
                            </div>
                        `;
                        grid.innerHTML += card;
                    });
                } catch (err) {
                    alert('Backend Connection Error');
                    loader.classList.add('hidden');
                }
            }
        </script>
    </body>
    </html>
    """

# --- PRODUCTION INFERENCE ENDPOINT ---
@app.get("/recommend/{user_raw_id}")
def get_recommendations(user_raw_id: int, k: int = 50):
    # 1. Transform Raw ID to Internal Encoding
    try:
        u_idx = metadata['user_encoder'].transform([user_raw_id])[0]
    except:
        return {"error": "User ID not found in training set"}

    # 2. Retrieval Stage: Candidate Generation
    with torch.no_grad():
        user_tensor = torch.tensor([u_idx])
        # Pass User ID through the User Tower (Neural Network)
        user_vec = model.user_net(model.user_embedding(user_tensor)).numpy()
        user_vec = user_vec.astype('float32') # FAISS requires float32

    # Query FAISS Index (ANN Search)
    # scores = dot product distance, indices = internal item indices
    scores, candidate_indices = index.search(user_vec, k)
    
    # 3. Ranking Stage: Metadata Mapping & Fine-Sorting
    # Convert internal indices back to original Movie/Content IDs
    retrieved_ids = metadata['item_encoder'].inverse_transform(candidate_indices[0])
    raw_scores = scores[0].tolist()

    # Apply the ranking logic (Fine-Ranking pass)
    # We combine them into a list of real-world results
    final_results = [
        {"id": int(rid), "score": float(rs)} 
        for rid, rs in zip(retrieved_ids, raw_scores)
    ]
    
    # Optionally sort by fine-ranker (simulated here)
    # final_results = ranker.rank(final_results)

    return {
        "user_id": user_raw_id,
        "results": final_results
    }