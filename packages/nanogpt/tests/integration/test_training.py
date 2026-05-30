"""Integration tests for training."""
import pytest
import torch
from nanogpt import GPT, ModelConfig


class TestTraining:
    """Integration tests for training."""

    def test_training_step_updates_parameters(self, small_model, dummy_batch):
        """Test that training step modifies model parameters."""
        x, y = dummy_batch

        # Get initial parameter values
        initial_params = {
            name: param.clone()
            for name, param in small_model.named_parameters()
        }

        # Training step
        optimizer = small_model.configure_optimizers(
            weight_decay=0.1,
            learning_rate=3e-4,
            device_type="cpu",
            verbose=False,
        )

        logits, loss = small_model(x, y)
        loss.backward()
        optimizer.step()

        # Check that parameters changed
        changed = False
        for name, param in small_model.named_parameters():
            if not torch.allclose(param, initial_params[name]):
                changed = True
                break

        assert changed, "Parameters should change after training step"

    def test_gradient_flow(self, small_model, dummy_batch):
        """Test that gradients flow through all parameters."""
        x, y = dummy_batch

        # Forward and backward
        logits, loss = small_model(x, y)
        loss.backward()

        # Check that all parameters have gradients
        params_without_grad = []
        for name, param in small_model.named_parameters():
            if param.requires_grad and param.grad is None:
                params_without_grad.append(name)

        assert len(params_without_grad) == 0, (
            f"Parameters without gradients: {params_without_grad}"
        )

    def test_multiple_training_steps(self, small_model, dummy_batch):
        """Test multiple training steps reduce loss."""
        x, y = dummy_batch

        optimizer = small_model.configure_optimizers(
            weight_decay=0.0,
            learning_rate=1e-3,
            device_type="cpu",
            verbose=False,
        )

        # Record initial loss
        with torch.no_grad():
            _, initial_loss = small_model(x, y)

        # Run 10 training steps
        for _ in range(10):
            optimizer.zero_grad()
            logits, loss = small_model(x, y)
            loss.backward()
            optimizer.step()

        # Check final loss
        with torch.no_grad():
            _, final_loss = small_model(x, y)

        # Loss should decrease (overfitting on this small batch)
        assert final_loss < initial_loss, "Loss should decrease after training"
