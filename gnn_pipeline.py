"""
gnn_pipeline.py v8.0 - External Feature Quality Improvements
────────────────────────────────────────────────────────────────────────────
การปรับปรุงจาก v7.0:
  1. ตัด total_excl_forestry ออก → ลด data leakage (target ≈ feature)
  2. เพิ่ม derived features จาก external:
       - transport_share     = energy_transport / (sum of sector cols)
       - forestry_growth     = delta forestry_land_use (leading indicator)
       - industrial_growth   = delta industrial_processes
       - transport_growth    = delta energy_transport
     → ลด multicollinearity, เพิ่ม predictive power
  3. Node-aware global injection:
       g_proj (B, N) ถูกปรับด้วย elec_weight (N,) per-node
       แทนที่จะ broadcast เท่ากันทุก node
       → GNN ได้ประโยชน์จาก spatial variation ของ external signal
  4. ปรับ global injection weight: 0.1 → 0.15
       เพราะ features คุณภาพดีขึ้น leakage หายไป
  5. เพิ่ม elec_weight normalizer ใน GraphWaveNet
────────────────────────────────────────────────────────────────────────────
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# 0. National External Features Loader + Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
def load_national_features(path: str = "./static/external.txt") -> pd.DataFrame:
    """
    โหลด external.txt และสร้าง derived features เพิ่ม:
      - transport_share:    สัดส่วน energy_transport ต่อ sector รวม
      - forestry_growth:    การเปลี่ยนแปลง forestry_land_use YoY
      - industrial_growth:  การเปลี่ยนแปลง industrial_processes YoY
      - transport_growth:   การเปลี่ยนแปลง energy_transport YoY
    """
    df = pd.read_csv(path)
    df["year"] = df["year"].astype(int)
    df = df.sort_values("year").reset_index(drop=True)

    # ── Derived: sector share (transport dominance) ──────────────────────────
    sector_cols = ["energy_transport", "industrial_processes", "agriculture", "waste"]
    sector_sum = df[sector_cols].sum(axis=1).replace(0, np.nan)
    df["transport_share"] = df["energy_transport"] / sector_sum

    # ── Derived: YoY growth rates (leading indicators) ───────────────────────
    df["forestry_growth"]   = df["forestry_land_use"].diff().fillna(0)
    df["industrial_growth"] = df["industrial_processes"].pct_change().fillna(0).clip(-2, 2)
    df["transport_growth"]  = df["energy_transport"].pct_change().fillna(0).clip(-2, 2)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 1. Graph Construction  (vectorized)
# ─────────────────────────────────────────────────────────────────────────────
def _haversine_matrix(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    R = 6371.0
    lat_r = np.radians(lats)
    lon_r = np.radians(lons)
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def build_adjacency_from_latlon(province_coords: dict, province_list: list,
                                 k: int = 5, threshold_km=None):
    n = len(province_list)
    lats = np.array([province_coords[p][0] for p in province_list])
    lons = np.array([province_coords[p][1] for p in province_list])
    dist_matrix = _haversine_matrix(lats, lons)
    np.fill_diagonal(dist_matrix, np.inf)

    src_list, dst_list, w_list = [], [], []
    for i in range(n):
        neighbors = np.argsort(dist_matrix[i])[:k]
        dists_i = dist_matrix[i][neighbors]
        finite_mask = dist_matrix[i] < np.inf
        sigma = dist_matrix[i][finite_mask].std()
        sigma = sigma if sigma > 0 else 1.0
        w_i = np.exp(-(dists_i ** 2) / (2 * sigma ** 2))
        if threshold_km is not None:
            keep = dists_i <= threshold_km
            neighbors, w_i = neighbors[keep], w_i[keep]
        src_list.extend([i] * len(neighbors))
        dst_list.extend(neighbors.tolist())
        w_list.extend(w_i.tolist())

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_weight = torch.tensor(w_list, dtype=torch.float)
    if edge_weight.numel() > 0 and edge_weight.max() > 0:
        edge_weight = edge_weight / edge_weight.max()

    return edge_index, edge_weight


def precompute_normalized_adj(num_nodes: int, edge_index: torch.Tensor,
                               edge_weight: torch.Tensor) -> torch.Tensor:
    adj = torch.zeros(num_nodes, num_nodes)
    adj[edge_index[0], edge_index[1]] = edge_weight
    d = adj.sum(dim=1)
    d_inv_sqrt = torch.pow(d + 1e-8, -0.5)
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Adaptive Adjacency
# ─────────────────────────────────────────────────────────────────────────────
class AdaptiveAdjacency(nn.Module):
    def __init__(self, num_nodes: int, embed_dim: int = 16):
        super().__init__()
        self.e1 = nn.Parameter(torch.empty(num_nodes, embed_dim))
        self.e2 = nn.Parameter(torch.empty(num_nodes, embed_dim))
        nn.init.xavier_uniform_(self.e1)
        nn.init.xavier_uniform_(self.e2)

    def forward(self):
        adj = F.relu(self.e1 @ self.e2.t())
        return adj / (adj.sum(dim=1, keepdim=True) + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Graph WaveNet  (Node-aware global injection)
# ─────────────────────────────────────────────────────────────────────────────
class GraphWaveNet(nn.Module):
    def __init__(self, num_nodes: int, in_channels: int, seq_len: int,
                 hidden_dim: int = 64, num_layers: int = 4, dropout: float = 0.4,
                 use_adaptive_adj: bool = True, n_global_feat: int = 0,
                 elec_weight: torch.Tensor = None):
        """
        elec_weight: (N,) tensor — per-node weight สำหรับ global injection
                     ถ้า None จะใช้ uniform weight (เหมือน v7)
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.use_adaptive_adj = use_adaptive_adj
        self.n_global_feat = n_global_feat

        if use_adaptive_adj:
            self.adaptive_adj = AdaptiveAdjacency(num_nodes, embed_dim=max(8, hidden_dim // 4))

        # Global context MLP: national features → (B, num_nodes)
        if n_global_feat > 0:
            self.global_proj = nn.Sequential(
                nn.Linear(n_global_feat, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_nodes),
            )
            # ── Node-aware weight (N,): เรียนรู้ว่า node ไหนควรรับ global signal มากแค่ไหน
            # init จาก elec_weight ถ้ามี (industrial nodes รับ signal มากกว่า)
            if elec_weight is not None:
                init_w = elec_weight.clone().float()
            else:
                init_w = torch.ones(num_nodes)
            # normalize → [0, 1]
            init_w = (init_w - init_w.min()) / (init_w.max() - init_w.min() + 1e-8)
            self.node_global_weight = nn.Parameter(init_w)  # (N,) — learnable

        self.start_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=(1, 1))

        self.filter_convs   = nn.ModuleList()
        self.gate_convs     = nn.ModuleList()
        self.batch_norms    = nn.ModuleList()
        self.residual_convs = nn.ModuleList()

        for i in range(num_layers):
            d = 2 ** i
            self.filter_convs.append(nn.Conv2d(hidden_dim, hidden_dim, (1, 2), dilation=d, padding=(0, d)))
            self.gate_convs.append(  nn.Conv2d(hidden_dim, hidden_dim, (1, 2), dilation=d, padding=(0, d)))
            self.batch_norms.append( nn.BatchNorm2d(hidden_dim))
            self.residual_convs.append(nn.Conv2d(hidden_dim, hidden_dim, (1, 1)))

        self.end_conv_1 = nn.Conv2d(hidden_dim, hidden_dim, (1, 1))
        self.end_conv_2 = nn.Conv2d(hidden_dim, 1, (1, 1))

        self.bias_layer = nn.Linear(num_nodes, num_nodes, bias=True)
        nn.init.eye_(self.bias_layer.weight)
        nn.init.zeros_(self.bias_layer.bias)
        with torch.no_grad():
            self.bias_layer.weight.mul_(0.95)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, static_adj: torch.Tensor,
                adp_adj=None, global_feat: torch.Tensor = None) -> torch.Tensor:
        """
        x:           (B, N, C, T)
        static_adj:  (N, N)
        global_feat: (B, n_global_feat) — national-level features (optional)
        """
        batch_size, num_nodes, _, _ = x.shape
        device = x.device

        x = x.permute(0, 2, 1, 3)   # (B, C, N, T)

        if self.use_adaptive_adj and adp_adj is not None:
            adj = 0.5 * static_adj + 0.5 * adp_adj
        else:
            adj = static_adj

        x = self.start_conv(x)
        skip = 0

        for i in range(len(self.filter_convs)):
            B, C, N, T = x.shape
            x_s = x.permute(0, 3, 1, 2).reshape(B * T, C, N)
            x_s = torch.bmm(x_s, adj.unsqueeze(0).expand(B * T, N, N))
            x_s = x_s.reshape(B, T, C, N).permute(0, 2, 3, 1)

            f  = torch.tanh(self.filter_convs[i](x_s))
            g  = torch.sigmoid(self.gate_convs[i](x_s))
            xt = self.dropout(f * g)
            xt = self.batch_norms[i](xt)

            xr     = self.residual_convs[i](x_s)
            min_t  = min(xt.shape[-1], xr.shape[-1])
            x      = xt[..., :min_t] + xr[..., :min_t]

            if i == 0:
                skip = xt
            else:
                skip = skip + xt[..., :skip.shape[-1]]

        out = F.relu(self.end_conv_1(F.relu(skip)))
        out = self.end_conv_2(out)   # (B, 1, N, T)
        out = out[:, 0, :, -1]      # (B, N)

        # ── Node-aware global injection ──────────────────────────────────────
        # v7: out = out + 0.1 * g_proj              ← broadcast เท่ากันทุก node
        # v8: out = out + 0.15 * (g_proj * node_w)  ← per-node weighting
        if self.n_global_feat > 0 and global_feat is not None:
            g_proj = self.global_proj(global_feat)                # (B, N)
            node_w = torch.sigmoid(self.node_global_weight)       # (N,) ∈ (0, 1)
            out = out + 0.15 * (g_proj * node_w.unsqueeze(0))    # per-node scale
        # ────────────────────────────────────────────────────────────────────

        out = self.bias_layer(out)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. Pinball / Quantile Loss
# ─────────────────────────────────────────────────────────────────────────────
class QuantileLoss(nn.Module):
    def __init__(self, q: float = 0.50):
        super().__init__()
        self.q = q

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        err = target - pred
        return torch.mean(torch.where(err >= 0, self.q * err, (self.q - 1) * err))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Dataset Preparation
# ─────────────────────────────────────────────────────────────────────────────
def prepare_stgnn_dataset(df: pd.DataFrame, province_list: list,
                           seq_len: int = 4, test_years: int = 3,
                           national_df: pd.DataFrame = None,
                           elec_df: pd.DataFrame = None) -> dict:
    df = df.copy()
    df["year"] = df["year"].astype(int)
    df = df[df["province"].isin(province_list)].copy()

    pivot = df.pivot(index="year", columns="province", values="CO2_tonnes")
    pivot = pivot.reindex(columns=province_list)
    if pivot.isnull().any().any():
        pivot = pivot.ffill().bfill()
    pivot = pivot.sort_index()

    years = sorted(pivot.index.tolist())
    data  = pivot.values     # (T, N)
    T, N  = data.shape

    if N != len(province_list):
        raise ValueError(f"Province mismatch: pivot={N}, list={len(province_list)}")

    scaler = StandardScaler()
    ds = scaler.fit_transform(data)   # (T, N) scaled

    # ─── Node-level temporal features ──────────────────────────────────────
    def lag(arr, k):
        out = np.zeros_like(arr)
        out[k:] = arr[:-k]
        out[:k] = arr[:k]
        return out

    lag1 = lag(ds, 1)
    lag2 = lag(ds, 2)
    lag3 = lag(ds, 3)

    cs = np.cumsum(ds, axis=0)
    roll3 = np.zeros_like(ds)
    roll3[0] = ds[0]
    roll3[1] = (ds[0] + ds[1]) / 2
    roll3[2:] = (cs[2:] - cs[:-2]) / 3

    growth = np.zeros_like(ds)
    growth[1:] = (ds[1:] - ds[:-1]) / (np.abs(ds[:-1]) + 1e-6)
    growth = np.clip(growth, -5, 5)

    year_idx = ((np.arange(T) / max(T - 1, 1)) * 2 - 1)[:, None] * np.ones((1, N))

    features = np.stack([ds, lag1, lag2, lag3, roll3, growth, year_idx],
                        axis=2).astype(np.float32)   # (T, N, 7)

    # ─── National external features (v8: ตัด leakage + เพิ่ม derived) ──────
    # ลบ total_excl_forestry ออก (leakage: ≈ target CO2)
    # เพิ่ม derived: transport_share, forestry_growth, industrial_growth, transport_growth
    EXT_COLS = [
        "energy_transport",
        "industrial_processes",
        "agriculture",
        "waste",
        "forestry_land_use",      # carbon sink — ค่าลบ = ดูด CO2
        "transport_share",        # NEW: สัดส่วน transport ต่อ sector รวม
        "forestry_growth",        # NEW: delta forestry YoY (leading)
        "industrial_growth",      # NEW: % change industrial (leading)
        "transport_growth",       # NEW: % change transport (leading)
    ]
    n_global_feat = 0
    global_seq_full = None

    if national_df is not None:
        available = [c for c in EXT_COLS if c in national_df.columns]
        if available:
            nat_indexed = national_df.set_index("year")
            ext_raw = (nat_indexed.reindex(years)[available]
                       .ffill().bfill().values.astype(np.float64))   # (T, n_ext)

            # ── ป้องกัน data leakage: fit scaler บน train เท่านั้น ──
            _unique_y   = sorted(df["year"].unique())
            _test_start = _unique_y[-test_years] if len(_unique_y) > test_years else _unique_y[0]
            _train_mask = np.array([y < _test_start for y in years])
            ext_scaler  = StandardScaler()
            ext_scaler.fit(ext_raw[_train_mask])
            ext_scaled  = ext_scaler.transform(ext_raw).astype(np.float32)

            global_seq_full = ext_scaled
            n_global_feat   = len(available)
            print(f"[GNN v8] External features ({n_global_feat}): {available}")

    # ─── Electricity static node features ──────────────────────────────────
    elec_static  = None
    elec_weight  = None   # (N,) สำหรับ node-aware global injection
    ELEC_FEAT_COLS = [
        "industrial_electricity", "residential_electricity",
        "public_electricity", "agriculture_electricity",
    ]
    if elec_df is not None:
        try:
            from process_elec import load_elec_profile, PROVINCE_TH_TO_EN
            elec_profile = load_elec_profile(elec_df)
            elec_arr = np.zeros((N, 4), dtype=np.float32)
            for j, prov in enumerate(province_list):
                if prov in elec_profile.index:
                    elec_arr[j] = elec_profile.loc[prov, ELEC_FEAT_COLS].values.astype(np.float32)
            elec_static = elec_arr  # (N, 4)

            # elec_weight = industrial_electricity (บอกว่า node ไหน "industrial" มากแค่ไหน)
            # → node ที่ industrial สูง = รับ national energy/transport signal มากกว่า
            elec_weight = torch.tensor(
                elec_arr[:, 0],   # industrial_electricity column
                dtype=torch.float
            )

            matched = int((elec_static != 0).any(axis=1).sum())
            print(f"[GNN v8] Elec static: {matched}/{N} provinces | node_weight from industrial_elec")
        except Exception as e:
            print(f"[GNN v8] Warning: elec_df ใส่ไม่ได้ — {e}")
            elec_static = None
            elec_weight = None

    # ถ้ามี elec_static → concat: (T, N, 7) → (T, N, 11)
    in_channels = 7
    if elec_static is not None:
        elec_broadcast = np.tile(elec_static[np.newaxis, :, :], (T, 1, 1))
        features = np.concatenate([features, elec_broadcast], axis=2).astype(np.float32)
        in_channels = 11
        print(f"[GNN v8] in_channels: 7 → 11 (+ 4 elec static)")

    # ─── Build sequences ────────────────────────────────────────────────────
    unique_years = sorted(df["year"].unique())
    test_start   = unique_years[-test_years] if len(unique_years) > test_years else unique_years[0]

    X, y, meta, G = [], [], [], []

    for t in range(seq_len, T):
        x_seq  = features[t - seq_len:t]
        x_tens = torch.tensor(x_seq.transpose(1, 2, 0), dtype=torch.float)
        X.append(x_tens)
        y.append(torch.tensor(ds[t], dtype=torch.float))
        meta.append(years[t])
        if global_seq_full is not None:
            G.append(torch.tensor(global_seq_full[t], dtype=torch.float))

    test_idx  = [i for i, yr in enumerate(meta) if yr >= test_start]
    train_idx = [i for i, yr in enumerate(meta) if yr <  test_start]
    if len(train_idx) < 5:
        split = int(len(X) * 0.8)
        train_idx = list(range(split))
        test_idx  = list(range(split, len(X)))

    def stack(indices):
        return torch.stack([X[i] for i in indices]), torch.stack([y[i] for i in indices])

    X_train, y_train = stack(train_idx)
    X_test,  y_test  = stack(test_idx)
    G_train = torch.stack([G[i] for i in train_idx]) if G else None
    G_test  = torch.stack([G[i] for i in test_idx])  if G else None

    return {
        "X_train": X_train, "y_train": y_train,
        "X_test":  X_test,  "y_test":  y_test,
        "G_train": G_train, "G_test": G_test,
        "meta_test": [meta[i] for i in test_idx],
        "scaler": scaler, "years": years,
        "seq_len": seq_len, "num_nodes": N, "in_channels": in_channels,
        "n_global_feat": n_global_feat,
        "full_features": features, "full_data_scaled": ds,
        "province_list": province_list,
        "global_seq_full": global_seq_full,
        "elec_static": elec_static,
        "elec_weight": elec_weight,   # NEW: (N,) tensor สำหรับ node-aware injection
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Training
# ─────────────────────────────────────────────────────────────────────────────
def train_gnn(dataset: dict, static_adj: torch.Tensor,
              num_nodes: int, hidden_dim: int = 64,
              epochs: int = 500, lr: float = 1e-3,
              dropout: float = 0.4, patience: int = 30,
              device: str = None) -> nn.Module:

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    static_adj = static_adj.to(device)

    X_all, y_all  = dataset["X_train"].to(device), dataset["y_train"].to(device)
    n_global_feat = dataset.get("n_global_feat", 0)
    G_all = dataset["G_train"].to(device) if dataset.get("G_train") is not None else None
    elec_weight = dataset.get("elec_weight")   # (N,) หรือ None

    val_size = max(1, int(0.15 * len(X_all)))
    X_val, y_val = X_all[-val_size:], y_all[-val_size:]
    X_tr,  y_tr  = X_all[:-val_size], y_all[:-val_size]
    G_val = G_all[-val_size:] if G_all is not None else None
    G_tr  = G_all[:-val_size] if G_all is not None else None

    if G_tr is not None:
        loader = DataLoader(TensorDataset(X_tr, y_tr, G_tr), batch_size=32, shuffle=True)
    else:
        loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)

    # ส่ง elec_weight เข้า model สำหรับ node-aware injection
    model = GraphWaveNet(
        num_nodes=num_nodes,
        in_channels=dataset["in_channels"],
        seq_len=dataset["seq_len"],
        hidden_dim=hidden_dim,
        num_layers=4,
        dropout=dropout,
        use_adaptive_adj=True,
        n_global_feat=n_global_feat,
        elec_weight=elec_weight,    # NEW
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2)
    criterion = QuantileLoss(q=0.50)

    best_loss, best_state, no_improve = float("inf"), None, 0

    for epoch in range(epochs):
        model.train()
        adp = model.adaptive_adj() if model.use_adaptive_adj else None
        total = 0.0

        for batch in loader:
            bx, by = batch[0], batch[1]
            bg = batch[2] if len(batch) == 3 else None
            optimizer.zero_grad()
            pred = model(bx, static_adj, adp, global_feat=bg)
            loss = criterion(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * bx.size(0)

        scheduler.step()

        model.eval()
        with torch.no_grad():
            adp = model.adaptive_adj() if model.use_adaptive_adj else None
            val_loss = criterion(model(X_val, static_adj, adp, global_feat=G_val), y_val).item()

        if val_loss < best_loss:
            best_loss  = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 100 == 0:
            avg = total / max(len(X_tr), 1)
            print(f"  Epoch {epoch+1:4d} | train={avg:.5f} | val={val_loss:.5f}")

        if no_improve >= patience:
            print(f"  Early stopping @ epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 7. Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_gnn(model: nn.Module, dataset: dict, static_adj: torch.Tensor,
                 province_list: list, device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    static_adj = static_adj.to(device)
    scaler = dataset["scaler"]
    G_test = dataset["G_test"].to(device) if dataset.get("G_test") is not None else None
    model.eval()

    with torch.no_grad():
        adp = model.adaptive_adj().to(device) if model.use_adaptive_adj else None
        preds_sc = model(dataset["X_test"].to(device), static_adj, adp,
                         global_feat=G_test).cpu().numpy()

    y_sc      = dataset["y_test"].numpy()
    preds_inv = scaler.inverse_transform(preds_sc)
    y_inv     = scaler.inverse_transform(y_sc)

    records = []
    for i, year in enumerate(dataset["meta_test"]):
        for j, prov in enumerate(province_list):
            records.append({
                "province": prov, "year": year,
                "CO2_tonnes": float(y_inv[i, j]),
                "preds":      float(max(preds_inv[i, j], 0)),
            })

    df_res = pd.DataFrame(records)
    mask   = df_res["CO2_tonnes"] > 0
    mape   = mean_absolute_percentage_error(
        df_res.loc[mask, "CO2_tonnes"], df_res.loc[mask, "preds"]) if mask.sum() else 0.0
    r2     = r2_score(df_res["CO2_tonnes"], df_res["preds"])
    print(f"[GNN v8] Eval → MAPE: {mape:.4f} | R²: {r2:.4f}")
    return df_res, mape, r2


# ─────────────────────────────────────────────────────────────────────────────
# 8. Predict next N years  (autoregressive)
# ─────────────────────────────────────────────────────────────────────────────
def gnn_predict_next_years(model: nn.Module, dataset: dict,
                            static_adj: torch.Tensor, province_list: list,
                            n_years: int = 1, device: str = None) -> pd.DataFrame:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    static_adj = static_adj.to(device)
    scaler          = dataset["scaler"]
    seq_len         = dataset["seq_len"]
    last_year       = max(dataset["years"])
    n_global_feat   = dataset.get("n_global_feat", 0)
    global_seq_full = dataset.get("global_seq_full")
    elec_static     = dataset.get("elec_static")

    feat_seq = dataset["full_features"][-seq_len:].copy()
    hist_sc  = dataset["full_data_scaled"][-seq_len:].copy()

    # extrapolate global_feat: linear trend จาก 3 ปีสุดท้าย
    future_global = []
    if global_seq_full is not None and n_global_feat > 0:
        last3    = global_seq_full[-3:]
        trend    = np.diff(last3, axis=0).mean(axis=0)
        last_val = global_seq_full[-1].copy()
        for step in range(n_years):
            last_val = last_val + trend
            future_global.append(last_val.copy())

    model.eval()
    all_preds = []

    with torch.no_grad():
        adp = model.adaptive_adj().to(device) if model.use_adaptive_adj else None

        for step in range(n_years):
            x_t = torch.tensor(feat_seq.transpose(1, 2, 0), dtype=torch.float).unsqueeze(0).to(device)
            gf  = None
            if future_global:
                gf = torch.tensor(future_global[step], dtype=torch.float).unsqueeze(0).to(device)

            pred_sc  = model(x_t, static_adj, adp, global_feat=gf).cpu().numpy()[0]
            pred_inv = np.maximum(scaler.inverse_transform(pred_sc.reshape(1, -1)).flatten(), 0)

            year_pred = last_year + step + 1
            for j, prov in enumerate(province_list):
                all_preds.append({"province": prov, "year": year_pred, "preds": float(pred_inv[j])})

            hist_sc = np.vstack([hist_sc[1:], pred_sc])
            t       = hist_sc.shape[0] - 1

            def safe_lag(k):
                return hist_sc[max(0, t - k)]

            roll3   = hist_sc[max(0, t - 2):t + 1].mean(axis=0)
            gr      = np.clip((hist_sc[t] - hist_sc[t - 1]) / (np.abs(hist_sc[t - 1]) + 1e-6), -5, 5)
            T_total = len(dataset["years"])
            yi      = ((T_total + step) / max(T_total, 1)) * 2 - 1

            new_f = np.stack([
                hist_sc[t], safe_lag(1), safe_lag(2), safe_lag(3),
                roll3, gr, np.full(len(province_list), yi),
            ], axis=1).astype(np.float32)

            if elec_static is not None:
                new_f = np.concatenate([new_f, elec_static], axis=1)

            feat_seq = np.vstack([feat_seq[1:], new_f[np.newaxis]])

    return pd.DataFrame(all_preds)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Ensemble Training
# ─────────────────────────────────────────────────────────────────────────────
def train_ensemble(dataset: dict, static_adj: torch.Tensor,
                   num_nodes: int, hidden_dim: int = 64,
                   epochs: int = 500, lr: float = 1e-3,
                   dropout: float = 0.4, patience: int = 30,
                   n_models: int = 5, device: str = None) -> list:
    models = []
    for i in range(n_models):
        print(f"[GNN v8] Training model {i+1}/{n_models}...")
        torch.manual_seed(42 + i)
        m = train_gnn(dataset, static_adj, num_nodes=num_nodes,
                      hidden_dim=hidden_dim, epochs=epochs, lr=lr,
                      dropout=dropout, patience=patience, device=device)
        models.append(m)
    return models


def ensemble_predict(models: list, x: torch.Tensor, static_adj: torch.Tensor,
                     device: str, global_feat: torch.Tensor = None) -> np.ndarray:
    preds = []
    for m in models:
        m.eval()
        with torch.no_grad():
            adp = m.adaptive_adj().to(device) if m.use_adaptive_adj else None
            gf  = global_feat.to(device) if global_feat is not None else None
            preds.append(m(x.to(device), static_adj, adp, global_feat=gf).cpu().numpy())
    return np.mean(preds, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Evaluate Ensemble
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_ensemble(models: list, dataset: dict, static_adj: torch.Tensor,
                      province_list: list, device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    static_adj = static_adj.to(device)
    scaler  = dataset["scaler"]
    G_test  = dataset.get("G_test")

    preds_sc  = ensemble_predict(models, dataset["X_test"], static_adj, device, global_feat=G_test)
    y_sc      = dataset["y_test"].numpy()
    preds_inv = scaler.inverse_transform(preds_sc)
    y_inv     = scaler.inverse_transform(y_sc)

    records = []
    for i, year in enumerate(dataset["meta_test"]):
        for j, prov in enumerate(province_list):
            records.append({
                "province": prov, "year": year,
                "CO2_tonnes": float(y_inv[i, j]),
                "preds":      float(max(preds_inv[i, j], 0)),
            })

    df_res = pd.DataFrame(records)
    mask   = df_res["CO2_tonnes"] > 0
    mape   = mean_absolute_percentage_error(
        df_res.loc[mask, "CO2_tonnes"], df_res.loc[mask, "preds"]) if mask.sum() else 0.0
    r2     = r2_score(df_res["CO2_tonnes"], df_res["preds"])
    print(f"[Ensemble v8] MAPE: {mape:.4f} | R²: {r2:.4f}")
    return df_res, mape, r2


def ensemble_predict_next_years(models: list, dataset: dict,
                                 static_adj: torch.Tensor, province_list: list,
                                 n_years: int = 1, device: str = None) -> pd.DataFrame:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    static_adj      = static_adj.to(device)
    scaler          = dataset["scaler"]
    seq_len         = dataset["seq_len"]
    last_year       = max(dataset["years"])
    n_global_feat   = dataset.get("n_global_feat", 0)
    global_seq_full = dataset.get("global_seq_full")
    elec_static     = dataset.get("elec_static")

    feat_seq = dataset["full_features"][-seq_len:].copy()
    hist_sc  = dataset["full_data_scaled"][-seq_len:].copy()

    future_global = []
    if global_seq_full is not None and n_global_feat > 0:
        last3    = global_seq_full[-3:]
        trend    = np.diff(last3, axis=0).mean(axis=0)
        last_val = global_seq_full[-1].copy()
        for step in range(n_years):
            last_val = last_val + trend
            future_global.append(last_val.copy())

    all_preds = []

    for step in range(n_years):
        x_t = torch.tensor(feat_seq.transpose(1, 2, 0), dtype=torch.float).unsqueeze(0)
        gf  = None
        if future_global:
            gf = torch.tensor(future_global[step], dtype=torch.float).unsqueeze(0)

        pred_sc  = ensemble_predict(models, x_t, static_adj, device, global_feat=gf)[0]
        pred_inv = np.maximum(scaler.inverse_transform(pred_sc.reshape(1, -1)).flatten(), 0)

        year_pred = last_year + step + 1
        for j, prov in enumerate(province_list):
            all_preds.append({"province": prov, "year": year_pred, "preds": float(pred_inv[j])})

        hist_sc  = np.vstack([hist_sc[1:], pred_sc])
        T_h      = hist_sc.shape[0]
        t        = T_h - 1
        roll3    = hist_sc[max(0, t - 2):t + 1].mean(axis=0)
        gr       = np.clip((hist_sc[t] - hist_sc[t - 1]) / (np.abs(hist_sc[t - 1]) + 1e-6), -5, 5)
        T_total  = len(dataset["years"])
        yi       = ((T_total + step) / max(T_total, 1)) * 2 - 1

        new_f = np.stack([
            hist_sc[t], hist_sc[max(0, t-1)], hist_sc[max(0, t-2)],
            hist_sc[max(0, t-3)], roll3, gr,
            np.full(len(province_list), yi)
        ], axis=1).astype(np.float32)

        if elec_static is not None:
            new_f = np.concatenate([new_f, elec_static], axis=1)

        feat_seq = np.vstack([feat_seq[1:], new_f[np.newaxis]])

    return pd.DataFrame(all_preds)


# ─────────────────────────────────────────────────────────────────────────────
# 11. Pipeline Runner
# ─────────────────────────────────────────────────────────────────────────────
def run_gnn_pipeline(
    df: pd.DataFrame,
    province_coords: dict,
    n_years: int = 1,
    k_neighbors: int = 3,
    seq_len: int = 4,
    hidden_dim: int = 64,
    epochs: int = 500,
    lr: float = 1e-3,
    dropout: float = 0.4,
    patience: int = 30,
    test_years: int = 3,
    device: str = None,
    use_ensemble: bool = True,
    n_models: int = 5,
    damping: float = 0.0,
    national_df: pd.DataFrame = None,
    elec_df: pd.DataFrame = None,
):
    province_list = sorted([p for p in province_coords if p in df["province"].values])
    if not province_list:
        raise ValueError("ไม่มีจังหวัดที่ตรงกันระหว่าง df และ province_coords")

    df = df[df["province"].isin(province_list)].copy()
    province_list = sorted(df["province"].unique().tolist())

    missing = [p for p in province_list if p not in province_coords]
    if missing:
        raise ValueError(f"Missing lat/lon: {missing}")

    print(f"[GNN v8] {len(province_list)} provinces | years: {df['year'].min()}–{df['year'].max()}")

    edge_index, edge_weight = build_adjacency_from_latlon(province_coords, province_list, k=k_neighbors)
    static_adj = precompute_normalized_adj(len(province_list), edge_index, edge_weight)

    dataset = prepare_stgnn_dataset(df, province_list,
                                     seq_len=seq_len, test_years=test_years,
                                     national_df=national_df,
                                     elec_df=elec_df)
    num_nodes = dataset["num_nodes"]

    if len(dataset["X_train"]) == 0:
        raise ValueError("ข้อมูลไม่เพียงพอสำหรับ GNN training")

    if use_ensemble:
        models = train_ensemble(
            dataset, static_adj, num_nodes=num_nodes,
            hidden_dim=hidden_dim, epochs=epochs, lr=lr,
            dropout=dropout, patience=patience,
            n_models=n_models, device=device)
        result_df, mape, r2 = evaluate_ensemble(models, dataset, static_adj, province_list, device=device)
        next_pred_df = ensemble_predict_next_years(
            models, dataset, static_adj, province_list, n_years=n_years, device=device)
    else:
        model = train_gnn(
            dataset, static_adj, num_nodes=num_nodes,
            hidden_dim=hidden_dim, epochs=epochs, lr=lr,
            dropout=dropout, patience=patience, device=device)
        result_df, mape, r2 = evaluate_gnn(model, dataset, static_adj, province_list, device=device)
        next_pred_df = gnn_predict_next_years(
            model, dataset, static_adj, province_list, n_years=n_years, device=device)

    historical_df = df[["province", "year", "CO2_tonnes"]].copy()
    historical_df = historical_df.rename(columns={"CO2_tonnes": "actual"})

    return result_df, next_pred_df, mape, r2, historical_df