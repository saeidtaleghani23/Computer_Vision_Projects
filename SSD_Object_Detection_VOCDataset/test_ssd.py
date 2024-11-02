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
        locs, class_scores = model(dummy_input)

    # Expected output shapes
    expected_locs_shape = (batch_size, 8732, 4)
    expected_class_scores_shape = (
        batch_size, 8732, config['dataset_params']['num_classes'])

    # Assert output shapes
    assert locs.shape == expected_locs_shape, f"Expected locs shape {expected_locs_shape}, but got {locs.shape}"
    assert class_scores.shape == expected_class_scores_shape, f"Expected class_scores shape {expected_class_scores_shape}, but got {class_scores.shape}"


# %%
if __name__ == "__main__":
    pytest.main()

# %%
