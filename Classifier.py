import os
import csv
from tqdm import tqdm
import torch
import argparse
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

class JesterDataset(Dataset):
    def __init__(self, root_dir,  split, transform=None):
        """
        Initialize the Jester dataset with the root directory for the images,
        the CSV file containing the labels, and an optional data transformation.

        Args:
            root_dir (str): Root directory for the Jester images.
            csv_file (str): Path to the CSV file containing the labels.
            split (str): Split to use ('train', 'val', or 'test').
            transform (callable, optional): Optional data transformation to apply to the images.
        """
        assert split in ['train', 'validation', 'test']
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.data = []
                    
        with open('jester-v1-' + self.split + '.csv') as f:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                video_dir = row[0]
                label = row[1]
                self.data.append((video_dir, label))

        self.label_dict = self._create_label_dict()

    def _create_label_dict(self):
        label_dict = {}
        with open('jester-v1-labels.csv', 'r') as f:
            reader = csv.reader(f)
            for idx, row in enumerate(reader):
                label_dict[row[0]] = idx
        return label_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_dir, label = self.data[idx]
        images = []
        for img_file in sorted(os.listdir(os.path.join(self.root_dir, video_dir))):
            img_path = os.path.join(self.root_dir, video_dir, img_file)
            image = Image.open(img_path)
            if self.transform:
                image = self.transform(image)
            images.append(image)
        images = torch.stack(images, dim=0)
        label = self.label_dict[label]
        return images, label

def collate_fn(batch):
    images, labels = zip(*batch)
    # Determine the maximum sequence length in the batch
    max_length = max([img.size(0) for img in images])
    print(max_length)
    # Pad sequences along the temporal dimension
    padded_images = []
    for img in images:
        padding = (0, 0, 0, 0, 0, 0, max_length - img.size(0), 0)  # Pad temporal dimension
        padded_img = F.pad(img, padding, "constant", 0)
        padded_images.append(padded_img)
    
    # Stack the padded images into a batch
    padded_images = torch.stack(padded_images, dim=0)
    labels = torch.tensor(labels)
    return padded_images, labels

class CNNLSTM(nn.Module):
    def __init__(self, num_classes=27):
        super(CNNLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.lstm = nn.LSTM(256 * 12 * 12, 512, batch_first=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.size()
        cnn_out = self.cnn(x.view(batch_size * seq_length, c, h, w))
        cnn_out = cnn_out.view(batch_size, seq_length, -1)
        lstm_out, _ = self.lstm(cnn_out)
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        return out

def evaluate(model, test_loader, criterion, device):
    model.eval()
    with torch.inference_mode():
        total_loss = 0.0
        num_correct = 0
        num_samples = 0

        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            _, predictions = torch.max(logits, dim=1)
            num_correct += (predictions == labels).sum().item()
            num_samples += len(inputs)

    avg_loss = total_loss / len(test_loader)
    accuracy = num_correct / num_samples

    return avg_loss, accuracy


def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, scheduler):
    model = model.to(device)
    for epoch in range(num_epochs):
        model.train()
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', position=0, leave=True) as pbar:
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = model(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

            avg_loss, accuracy = evaluate(model, val_loader, criterion, device)
            print(f'Validation set: Average loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}')

        scheduler.step()


def test(model, test_loader, device):
    model = model.to(device)
    model.eval()
    with torch.inference_mode():
        all_preds = []
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            _, predictions = torch.max(logits, dim=1)
            preds = list(zip(labels, predictions.tolist()))
            all_preds.extend(preds)
    return all_preds


def main(args):
    image_net_mean = torch.Tensor([0.485, 0.456, 0.406])
    image_net_std = torch.Tensor([0.229, 0.224, 0.225])

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((100, 176)),
        transforms.Normalize(image_net_mean, image_net_std),
    ])

    data_root = r'/home/enricus/Lib/uni/year III/sem I/computer vision/a3/CV-3/20bn-jester-v1'

    train_dataset = JesterDataset(data_root, 'train', transform=data_transform)
    val_dataset = JesterDataset(data_root, 'validation', transform=data_transform)

    batch_size = 32
    num_workers = 4

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # model = MyConv(num_classes=len(train_dataset.label_dict))
    model = CNNLSTM(num_classes=len(train_dataset.label_dict))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    if not args.test:
        train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=5, scheduler=scheduler)
        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, 'model2.ckpt')
    else:
        test_dataset = val_dataset
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        checkpoint = torch.load(args.checkpoint, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        preds = test(model, test_loader, device)
        write_predictions(preds, 'predictions.csv')

def write_predictions(preds, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for im, pred in preds:
            writer.writerow((im, pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--checkpoint', default='model.ckpt')
    args = parser.parse_args()
    main(args)
    
    
#               python Classifier.py 