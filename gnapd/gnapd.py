import torch
import torch.nn as nn
from GNAN import GNAN, TensorGNAN
from torch_geometric.data import Data
import torch_geometric as pyg

def main():

    torch.random.manual_seed(42)

    num_params = 2**10
    dummy_params = torch.randn(num_params) * 10


    ## DECOMPOSE
    num_components = 8
    components = torch.randn((num_components, num_params))  # Nodes
    components = nn.Parameter(components)
    print("Components shape:", components.shape)
    edge_index = [[i for i in range(num_components) for j in range(num_components)],
                  [j for i in range(num_components) for j in range(num_components)]]  # Fully connected graph
    edge_index = torch.tensor(edge_index)
    print(edge_index)
    print("Edge index shape:", edge_index.shape)
    edge_weight = torch.tensor([1 if i != j else 0 for i,j in zip(edge_index[0], edge_index[1])], dtype=torch.float)
    # laplacean normalization
    # edge_index, edge_weight = pyg.utils.get_laplacian(edge_index, num_nodes=num_components, normalization="sym")
    print(edge_weight)

    print('___')
    value, count = torch.unique(edge_weight, return_counts=True, sorted=True)
    print(value, count)
    distance_count = torch.ones((num_components, num_components))
    for edge in edge_index.t():
        if edge[0] == edge[1]:
            # Value 0 = # of self-loops
            distance_count[edge[0], edge[1]] = count[0].item()
        else:
            # Value 1 = # of non-self-loops
            distance_count[edge[0], edge[1]] = count[1].item()

    norm_matrix = 1 / distance_count
    print(norm_matrix)
    # input()

    data = Data(x=components, edge_index=edge_index, node_distances=edge_weight, normalization_matrix=norm_matrix)

    model = TensorGNAN(
        in_channels=num_params,
        out_channels=1,
        n_layers=1,
        hidden_channels=None,
        bias=True,
        dropout=0.0,
        device='cpu',
        normalize_rho=False,
        rho_per_feature=False
    )
    print("Model", model, sep="\n")

    num_epochs = 1000
    optimizer = torch.optim.AdamW(list(model.parameters()) + [components], lr=0.01)
    criterion = nn.MSELoss()
    for i in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        coefficients = model(data)
        # print("Coefficients shape:", coefficients.shape)
        print("Coefficients:", coefficients)
        # print("Components shape:", components.shape)
        out = torch.matmul(coefficients.T, components)
        loss = criterion(out, dummy_params)
        print(f"Epoch {i}: loss={loss.item()}")
        loss.backward()
        optimizer.step()

if __name__ == "__main__":

    main()