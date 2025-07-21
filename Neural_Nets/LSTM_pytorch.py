import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    A simple LSTM-based neural network for sequence modeling.

    Args:
        input_dim (int): Number of expected features in the input.
        hidden_dim (int): Number of features in the hidden state.
        output_dim (int): Number of output features.
        num_layers (int): Number of recurrent layers (default: 1).
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layer: batch_first=True expects input shape (batch, seq, feature)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Fully connected layer to map hidden state to output
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, input_dim)
                        or (batch_size, input_dim) for a single timestep.

        Returns:
            Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # Ensure input has 3 dimensions (batch, seq, feature)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        batch_size = x.size(0)

        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)

        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))  # out: (batch, seq, hidden_dim)

        # Use the hidden state from the last time step
        out = self.fc(out[:, -1, :])  # (batch, output_dim)
        return out

if __name__ == "__main__":
    # Example usage and testing

    # Parameters
    input_dim = 10      # Number of input features
    hidden_dim = 20     # Number of hidden units
    output_dim = 5      # Number of output features
    num_layers = 2      # Number of LSTM layers
    batch_size = 4      # Number of samples in a batch
    seq_length = 7      # Length of the input sequence

    # Create a random input tensor
    x = torch.randn(batch_size, seq_length, input_dim)

    # Instantiate the model
    model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers)

    # Forward pass
    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Output:", output)