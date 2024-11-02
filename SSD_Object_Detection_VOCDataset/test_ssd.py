import torch
import yaml
import pytest
from model import SSD  # Replace with the actual path if different

# Load configuration


@pytest.fixture
def config():
    with open("./config/config.yml", "r") as file:
        return yaml.safe_load(file)

# Initialize the model


@pytest.fixture
def model(config):
    return SSD(config)


def test_ssd_forward_pass(model, config):
    # Create a dummy input tensor with batch size 2 and dimensions (3, 300, 300)
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 300, 300)

    # Run forward pass
    with torch.no_grad():  # Disable gradient calculations
        losses, detections = model(dummy_input)
        print(f'losses: {losses}')
        print(f'detections: {detections}')


# %%
if __name__ == "__main__":
    pytest.main()

# %%
