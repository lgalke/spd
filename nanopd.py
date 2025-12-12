

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import random
import numpy as np
import torch



class Components(nn.Module):
    def __init__(self, model: nn.Module, n_components:int):
        super(Components, self).__init__()
        all_layers = []
        indices = []
        sizes = []
        for idx, weight in enumerate(model.parameters()):
            if weight.ndim == 2:
                all_layers.append(weight)
                indices.append(idx)
                sizes.append(weight.size())
        all_layers_stacked = torch.cat([l.flatten() for l in all_layers])
        self.components = nn.Parameter(torch.zeros((n_components, all_layers_stacked.size(0))), requires_grad=True)
        self.model = model

        # Init params
        for i in range(n_components):
            self.components.data[i, :] = all_layers_stacked + 0.01 * torch.randn_like(all_layers_stacked)

    def apply_components_to_model(self, mask=None):
        if mask is None:
            return self.components
        else:
            return self.components[mask, :]


    def forward(self):
        return self.components

def decompose(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
              n_components:int=10):
    """Decomposes a BitLinear model into its binary components."""

    components = Components(model, n_components=n_components)
    print("Components size:", components.size()) # n_components x n_weights

    ## Init with slightly noisy average of all weights

    with torch.no_grad():
        print("mean @ init", components.mean())
        print("std @ init", components.std())


    for batch in train_loader:
        x, y = batch
        print("x", x)
        print("y", y)
        break




def main():
    # Set seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    n = 10000
    n_train = 5000
    n_test = n - n_train
    n_epochs = 1000
    bsz = 100
    break_on_first_solution = False

    lr = 0.1

    n_hidden = 8  # BitLinear seems to need more than Linear
    use_bitlinear = True  # change to false for basic MLP
    weight_measure = "AbsMedian"  # AbsMean, AbsMedian
    bitlinear_bias = False

    # Build toy dataset
    x = torch.rand(n, 4) < 0.5
    y = torch.logical_xor(x[:, 0], x[:, 1]).long()
    x = x.float()

    print("Y mean (should be about 0.5):", y.float().mean())

    # split..
    x_train, y_train = x[:n_train], y[:n_train]
    x_test, y_test = x[n_train:], y[n_train:]

    # Basic MLP
    model = nn.Sequential(
        nn.Linear(4, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, 2),
    )

    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(list(zip(x_train, y_train)), shuffle=True, batch_size=bsz)

    for i in range(n_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            x, y = batch
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            print(loss.item())

        # eval
        model.eval()
        y_hat = model(x_test)
        acc = (torch.argmax(y_hat, dim=-1) == y_test).sum().float() / y_test.size(0)
        print(f"Epoch {i} Accuracy: {acc.item()}")
        if break_on_first_solution and (acc - 1.0).abs() < 1e-8:
            break


    decompose(model, train_loader, None, n_components=10)


    exit(0)

    with torch.no_grad():
        # Check if inputs 2-3 are ignored
        print("Model", model, sep="\n")
        print("Model weights:")
        for i, m in enumerate(model.modules()):
            print(m)
            if not isinstance(m, BitLinear):
                print("Skipping.")
                continue
            # print("BitLinear shadow weights", m.weight, sep="\n")
            w_quant = quantize_weights(m)
            print("BitLinear quantized weights", w_quant, sep="\n")
            print("w_quant.size()", w_quant.size())

        print("-" * 80)
        print("Analyzing input2hidden layer...", model[0])
        i2h_w = (
            quantize_weights(model[0])
            if isinstance(model[0], BitLinear)
            else model[0].weight
        )
        print(
            "Input2hidden weights cols 0-1 (should be xor)", i2h_w[:, [0, 1]], sep="\n"
        )
        print(
            "Input2hidden weights cols 0-1 L1 norm:",
            torch.linalg.vector_norm(i2h_w[:, [0, 1]], 1).item(),
        )
        print("-" * 80)
        print(
            "Input2hidden weights cols 2-3 (should be zero)", i2h_w[:, [2, 3]], sep="\n"
        )
        print(
            "Input2hidden weights cols 2-3 L1 norm:",
            torch.linalg.vector_norm(i2h_w[:, [2, 3]], 1).item(),
        )
        print("-" * 80)
        print("Analyzing input2hidden bias...", model[0].bias)
        print("-" * 80)

        print("-" * 80)
        print("Analyzing hidden2output layer...", model[2])

        h2i_w = (
            quantize_weights(model[-1])
            if isinstance(model[-1], BitLinear)
            else model[-1].weight
        )
        print("hidden2output weights", h2i_w, sep="\n")
        print(
            "hidden2output weights L1 norm:", torch.linalg.vector_norm(h2i_w, 1).item()
        )
        print("-" * 80)
        print("Analyzing hidden2output bias...", model[2].bias)
        print("-" * 80)


if __name__ == "__main__":
    main()


