import sys, os
sys.path.insert(0, os.path.abspath("."))
import torch
import numpy as np
from ai.intersection_ai.model import IntersectionTrajectoryMLP

sd = torch.load('models/intersection/trajectory_model.pth', weights_only=True, map_location='cpu')
m = IntersectionTrajectoryMLP(40, 64, 12)
m.load_state_dict(sd)
m.eval()

np.random.seed(42)
x_np = np.random.randn(1, 40).astype(np.float32)
x_th = torch.tensor(x_np)
with torch.no_grad():
    out_th = m(x_th).numpy()

w0 = sd['net.0.weight'].numpy()
b0 = sd['net.0.bias'].numpy()
gamma = sd['net.1.weight'].numpy()
beta = sd['net.1.bias'].numpy()
w1 = sd['net.3.weight'].numpy()
b1 = sd['net.3.bias'].numpy()
w2 = sd['net.5.weight'].numpy()
b2 = sd['net.5.bias'].numpy()

h1 = x_np @ w0.T + b0
mean = h1.mean(axis=-1, keepdims=True)
# PyTorch LayerNorm uses biased variance (ddof=0) and eps=1e-5
var = h1.var(axis=-1, keepdims=True)
h1 = (h1 - mean) / np.sqrt(var + 1e-5) * gamma + beta
h1 = np.maximum(h1, 0)
h2 = np.maximum(h1 @ w1.T + b1, 0)
out_np = h2 @ w2.T + b2

diff = np.max(np.abs(out_th - out_np))
print(f"Max diff between PyTorch and NumPy: {diff:.8e}")
assert diff < 1e-5, f"Diff too large: {diff}"
print("Equivalence verified successfully!")
