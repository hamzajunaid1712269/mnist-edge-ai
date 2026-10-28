from torchvision.datasets import MNIST

dataset = MNIST(
    root="./data",
    train=False,
    download=True
)

image, label = dataset[0]

image.save("test_digit.png")

print("Saved test_digit.png")
print("True label:", label)