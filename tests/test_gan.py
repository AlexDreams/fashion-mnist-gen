import torch

from fashion_mnist_gen.gan import ConditionalDiscriminator, ConditionalGenerator


def test_conditional_gan_outputs_images_and_real_fake_scores() -> None:
    generator = ConditionalGenerator(noise_dim=16)
    discriminator = ConditionalDiscriminator()
    labels = torch.tensor([0, 4, 9])
    noise = torch.randn(3, 16)

    images = generator(noise, labels)
    scores = discriminator(images, labels)

    assert images.shape == (3, 1, 28, 28)
    assert torch.all((images >= 0) & (images <= 1))
    assert scores.shape == (3,)


def test_conditional_gan_sampling_is_reproducible() -> None:
    generator = ConditionalGenerator(noise_dim=16).eval()
    labels = torch.tensor([2, 5, 8])
    first_seed = torch.Generator().manual_seed(123)
    second_seed = torch.Generator().manual_seed(123)

    first_samples = generator.sample(labels, generator=first_seed)
    second_samples = generator.sample(labels, generator=second_seed)

    assert torch.equal(first_samples, second_samples)


def test_gan_generator_and_discriminator_have_gradients() -> None:
    generator = ConditionalGenerator(noise_dim=8)
    discriminator = ConditionalDiscriminator()
    labels = torch.tensor([1, 7])
    images = generator(torch.randn(2, 8), labels)
    loss = discriminator(images, labels).sum()
    loss.backward()

    assert generator.project[0].weight.grad is not None
    assert discriminator.score.weight.grad is not None


def test_collapse_monitor_reports_diversity_and_flags_collapse() -> None:
    from train_gan import CollapseMonitor

    real = [torch.rand(6, 1, 28, 28, generator=torch.Generator().manual_seed(c)) for c in range(10)]
    monitor = CollapseMonitor(
        real, noise_dim=8, seed=0, device=torch.device("cpu"),
        classifier=None, collapse_ratio=0.6,
    )
    generator = ConditionalGenerator(noise_dim=8)
    with torch.no_grad():
        # Ignore the noise entirely: every sample of a class becomes identical.
        generator.project[0].weight[:, :8].zero_()

    result = monitor(generator)

    assert result["min_pixel_diversity_ratio"] < 0.6
    assert len(result["collapsed_classes"]) == 10
    assert generator.training
