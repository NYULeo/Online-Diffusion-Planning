from adjoint_matching import train_adjoint_matching
from Pretrain.Rewards.nets import ScalarReward
from Pretrain.Planners.Backbone.UNet import TemporalUnet
import torch


if __name__ == "__main__":
    # Example usage of the Adjoint Matching training without a dataset.
    # In practice, replace the reward and backbone initialisations with
    # loading of your pretrained models (e.g. via torch.load).
    horizon = 10
    state_dim = 4
    action_dim = 2

    # Instantiate a dummy reward network (Beta distribution model) and backbone.
    reward_net = ScalarReward(obs_dim=state_dim, act_dim=action_dim)
    backbone = TemporalUnet(horizon=horizon, transition_dim=state_dim + action_dim)

    # Train the control network via Adjoint Matching without a dataset.
    trained_control = train_adjoint_matching(
        horizon=horizon,
        state_dim=state_dim,
        action_dim=action_dim,
        reward_net=reward_net,
        backbone=backbone,
        num_iterations=3,
        batch_size=2,
        lr=5e-4,
    )

    # Save the trained control network if desired
    torch.save(trained_control.state_dict(), "control_net_no_dataset.pt")