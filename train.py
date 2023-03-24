import torch
from unet import UNet
from utils import load_data_set
from torchvision.transforms import transforms
from pathlib import Path
import torchvision

config = {
    "lr": 1e-3,
    "batch_size": 32,
    "image_dir": "data/converted_img",
    "masks_dir": "data/mask",
    "epochs": 16,
    "checkpoint_path": "checkpoint/rocks_segm_v1.pth",
    "optimiser_path": "checkpoint/rocks_segm_optim_v1.pth",
    "continue_train": False,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


transforms_image = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0., 0., 0.), (1., 1., 1.))
])

transforms_mask = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.,), (1.,))
])

train_dataset, val_dataset = load_data_set(
    config['image_dir'],
    config['masks_dir'],
    transforms=[transforms_image, transforms_mask],
    batch_size=config['batch_size']
)

def val_loop(model, optimiser, epoch, val_dataset):
    torch.save(model.state_dict(), config["checkpoint_path"])
    torch.save(optimiser.state_dict(), config["optimiser_path"])
    model.eval()

    num_correct = 0
    num_pixel = 0
    dice_score = 0

    with torch.no_grad():
        for x, y in val_dataset:
            x, y = x.to(config["device"]), y.to(config["device"])
            preds = model(x)
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixel += torch.numel(preds)
            dice_score += (2*(preds*y).sum()) / ((preds + y).sum() + 1e-8)

            torchvision.utils.save_image(preds, f"test/pred/{epoch}.png")
            torchvision.utils.save_image(y, f"test/true/{epoch}.png")
            torchvision.utils.save_image(x, f"test/orig/{epoch}.png")

    print(f"Dice score = {dice_score/len(val_dataset)}")
    model.train()

def train_loop(model, train_dataloader, val_dataloader, optimiser, loss_fn, epochs, device, accuracy_fn = None):

    train_loss, train_acc = 0, 0
    model.to(device)
    for epoch in range(epochs):
        for (X, y) in (train_dataloader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            loss = loss_fn(y_pred, y)
            train_loss += loss
            # train_acc += accuracy_fn(y_true=y, y_pred=y_pred)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            train_loss /= len(train_dataloader)
            train_acc /= len(train_dataloader)
        val_loop(model, optimiser, epoch, val_dataloader)



if __name__ == "__main__":
    model = UNet(3)
    optimiser = torch.optim.Adam(params=model.parameters(), lr=config['lr'])
    loss_fn = torch.nn.BCELoss()
    model.train()
    train_loop(model=model, train_dataloader=train_dataset,val_dataloader=val_dataset, optimiser=optimiser,loss_fn=loss_fn, epochs=config["epochs"],device=config["device"] )
    print("Finished!")