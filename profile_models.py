import torch
import time
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool

# === Mock Classes for Dependency Resolution ===

class MockEncoding:
    def __init__(self):
        self.idx2type = list(range(5))
        self.idx2table = list(range(10))
        self.idx2join = list(range(5))
        self.idx2op = list(range(5))
        self.idx2col = list(range(20))

class MockFeatureEmbed(nn.Module):
    def __init__(self, embed_size, **kwargs):
        super().__init__()
        self.embed_size = embed_size
        self.out_dim = 64 * 5 + 64 // 8 + 1 # From GNTO_QF definition
    
    def forward(self, x):
        # Mock output based on input shape
        # Assuming x is [num_nodes, feat_dim] but we just return random embeddings
        return torch.randn(x.size(0), self.out_dim).to(x.device)

# === Reconstruct GNTO_QF Model (Simplified for Profiling) ===
class GNTO_QF_Profile(nn.Module):
    def __init__(self, encoding, hidden_dim=64):
        super().__init__()
        self.node_encoder = MockFeatureEmbed(embed_size=64)
        self.input_dim = self.node_encoder.out_dim
        self.gnn = GATConv(self.input_dim, hidden_dim, heads=4, concat=False)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, data):
        x = self.node_encoder(data.x)
        x = self.gnn(x, data.edge_index)
        x = torch.relu(x)
        x = global_mean_pool(x, data.batch)
        return self.head(x)

# === Mock QueryFormer (Approximate Structure) ===
# Assuming standard Transformer Encoder structure
class QueryFormer_Profile(nn.Module):
    def __init__(self, embed_size=64, n_layers=8, ffn_dim=128, head_size=8):
        super().__init__()
        self.node_encoder = MockFeatureEmbed(embed_size=embed_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=head_size, dim_feedforward=ffn_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(embed_size, 1)
        self.embed_size = embed_size

    def forward(self, data):
        # Transformer expects [batch, seq_len, dim]
        # We need to pad/mask properly for real usage, but for profiling FLOPs/Time:
        # We will treat all nodes as a single sequence per graph or similar
        
        x = self.node_encoder(data.x) # [Total_Nodes, Dim] -> [Total_Nodes, Out_Dim]
        # Project to embed_size for Transformer
        if x.size(1) != self.embed_size:
            # Add a projection if dimensions don't match (mocking real behavior)
            self.proj = nn.Linear(x.size(1), self.embed_size).to(x.device)
            x = self.proj(x)

        # Reshape for transformer: [Batch_Size, Max_Nodes, Dim]
        # For simplicity in profiling, we just reshape to [Batch, Nodes_Per_Graph, Dim]
        # assuming uniform graph size for this test
        batch_size = data.batch.max().item() + 1
        nodes_per_graph = x.size(0) // batch_size
        x = x.view(batch_size, nodes_per_graph, -1)
        
        out = self.transformer(x)
        out = out.mean(dim=1) # Pool
        return self.head(out)

def profile_model(model, name, device, batch_size=128, num_nodes=50, repeats=100):
    model = model.to(device)
    model.eval()
    
    # Generate Mock Data
    # Batch of 128 graphs, each with ~50 nodes
    total_nodes = batch_size * num_nodes
    x = torch.randn(total_nodes, 10) # Dummy input features
    edge_index = torch.randint(0, total_nodes, (2, total_nodes * 2)) # Random edges
    batch = torch.arange(batch_size).repeat_interleave(num_nodes)
    
    data = Data(x=x, edge_index=edge_index, batch=batch).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(data)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    
    for _ in range(repeats):
        with torch.no_grad():
            _ = model(data)
            
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()
    
    avg_time = (end - start) / repeats * 1000 # ms
    print(f"{name}: {avg_time:.2f} ms per batch")
    
    # Count params
    params = sum(p.numel() for p in model.parameters())
    print(f"{name} Params: {params:,}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Profiling on {device}...")
    
    encoding = MockEncoding()
    
    # 1. GNTO_QF Profile
    gnto = GNTO_QF_Profile(encoding, hidden_dim=64)
    profile_model(gnto, "GNTO_QF (Approx)", device)
    
    # 2. QueryFormer Profile (Using PyTorch Transformer to simulate heavy compute)
    # Parameters tuned to match ~4.5M params
    # embed_size=64, layers=8, ffn=128 -> this is small
    # To get ~4.5M, let's increase dimensions
    qf = QueryFormer_Profile(embed_size=512, n_layers=6, ffn_dim=2048, head_size=8)
    profile_model(qf, "QueryFormer (Simulated Large)", device)

if __name__ == "__main__":
    main()

