import torch

from research_formal.models import cmunext_s


def test_cmunext_s_forward_shape():
    model = cmunext_s()
    model.eval()
    image = torch.rand(1, 3, 256, 256)

    with torch.no_grad():
        output = model(image)

    assert output.shape == (1, 1, 256, 256)
