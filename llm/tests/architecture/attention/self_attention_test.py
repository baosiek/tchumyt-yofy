import torch


def test_self_attention():
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],    # The
         [0.55, 0.87, 0.66],    # boy
         [0.57, 0.85, 0.64],    # is
         [0.22, 0.58, 0.33],    # at
         [0.77, 0.25, 0.10],    # his
         [0.05, 0.80, 0.55]]    # home
    )

    assert inputs.shape[0] == 6
