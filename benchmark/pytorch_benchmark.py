from sklearn.datasets import fetch_openml
import torch
import torch.nn as nn
import numpy as np

# Same data loading as your main.py
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X = mnist.data.astype(np.float32)
Y = mnist.target.astype(np.int64)


X_TRAIN = torch.tensor(X[:55000] / 255, dtype=torch.float32)
X_VAL   = torch.tensor(X[55000:60000] / 255, dtype=torch.float32)
X_TEST  = torch.tensor(X[60000:] / 255, dtype=torch.float32)
Y_TRAIN = torch.tensor(Y[:55000], dtype=torch.long)
Y_VAL   = torch.tensor(Y[55000:60000], dtype=torch.long)
Y_TEST  = torch.tensor(Y[60000:], dtype=torch.long)

model = nn.Sequential(
    nn.Linear(784, 128), nn.ReLU(),
    nn.Linear(128, 128), nn.ReLU(),
    nn.Linear(128, 10)
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

batch_size = 128
initial_lr = 0.01
decay_rate = 0.4
decay_steps = 10

best_val_acc = 0
patience, min_delta = 5, 0.0005
epochs_without_improvement = 0

for epoch in range(100):
    # Learning rate decay, matching the NumPy version's schedule
    if epoch > 0 and epoch % 2 == 0:
        lr = initial_lr * decay_rate ** (epoch / decay_steps)
        for g in optimizer.param_groups:
            g['lr'] = lr

    perm = torch.randperm(len(X_TRAIN))
    for i in range(0, len(X_TRAIN), batch_size):
        idx = perm[i:i+batch_size]
        optimizer.zero_grad()
        out = model(X_TRAIN[idx])
        loss = loss_fn(out, Y_TRAIN[idx])
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        val_acc = (model(X_VAL).argmax(1) == Y_VAL).float().mean().item()
    print(f"Epoch {epoch+1}: val accuracy = {val_acc*100:.2f}% (lr={optimizer.param_groups[0]['lr']:.6f})")

    if val_acc > best_val_acc + min_delta:
        best_val_acc = val_acc
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
    if epochs_without_improvement >= patience:
        print("Plateau reached, stopping")
        break

with torch.no_grad():
    test_acc = (model(X_TEST).argmax(1) == Y_TEST).float().mean().item()
print(f"\nFinal test accuracy: {test_acc*100:.2f}%")