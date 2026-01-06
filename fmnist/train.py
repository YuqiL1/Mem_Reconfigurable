import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd

from NDR import NDRActivation

from torch.autograd import Function

class STEQuantize(Function):
    @staticmethod
    def forward(ctx, input, num_bits):
        qmax = 2 ** (num_bits - 1) - 1
        qmin = -qmax
        max_val = input.abs().max()
        scale = max_val / qmax if max_val != 0 else 1.0
        input_int = torch.clamp((input / scale).round(), qmin, qmax)
        ctx.scale = scale
        ctx.save_for_backward(input)
        return input_int * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def ste_quantize(x, num_bits):
    return STEQuantize.apply(x, num_bits)

class QuantLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, num_bits=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_bits = num_bits

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in**0.5 if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        quant_weight = ste_quantize(self.weight, self.num_bits)
        if not self.training:
            print(f"Quantized weight sample (first 10): {quant_weight.flatten()[:10].cpu().numpy()}")
        return nn.functional.linear(input, quant_weight, self.bias)

class ActSTEQuant(nn.Module):
    """Activation fake-quantization with STE."""
    def __init__(self, num_bits: int | None):
        super().__init__()
        self.num_bits = num_bits

    def forward(self, x):
        if self.num_bits is None:
            return x
        return ste_quantize(x, self.num_bits)

class ConfigurableMLP(nn.Module):
    def __init__(self, activation: nn.Module, hidden_dim=400, w_bits=3, a_bits_in=8, a_bits_out=8):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = QuantLinear(28*28, hidden_dim, num_bits=w_bits)
        self.act1 = activation
        self.fc2 = nn.Linear(hidden_dim, 10)
        self.q_in = ActSTEQuant(a_bits_in)
        self.q_out = ActSTEQuant(a_bits_out)

    def forward(self, x):
        x = self.flatten(x)
        x = self.q_in(x)
        x = self.act1(self.fc1(x))
        x = self.q_out(x)
        return self.fc2(x)

def train_and_eval(model, train_loader, test_loader, device,
                   epochs=50, lr=1e-3, save_path='best_qat.pth'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    init_snap = save_path.replace('.pth', '_epoch0.pth')
    torch.save(model.state_dict(), init_snap)
    print(f"  >>> Initial snapshot saved at epoch 0 to '{init_snap}'")

    best_acc = 0.0
    history = {
        'train_loss': [],
        'test_acc'  : []
    }

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            total_samples += imgs.size(0)

        train_loss = running_loss / total_samples
        history['train_loss'].append(train_loss)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        test_acc = 100.0 * correct / total
        history['test_acc'].append(test_acc)

        print(f"Epoch {epoch}/{epochs}  "
              f"Train Loss: {train_loss:.4f}  "
              f"Test Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            ckpt = {
                'model_state': model.state_dict(),
                'in_alpha': model.act1.in_alpha.detach().cpu(),
                'in_beta' : model.act1.in_beta.detach().cpu(),
                'alpha'   : model.act1.alpha.detach().cpu(),
                'beta'    : model.act1.beta.detach().cpu(),
            }
            torch.save(ckpt, save_path)
            print(f"  >>> New best acc: {best_acc:.2f}%, saved to '{save_path}'")

        if epoch in {5, 10, 20, epochs}:
            snap_name = save_path.replace('.pth', f'_epoch{epoch}.pth')
            torch.save(model.state_dict(), snap_name)
            print(f"  >>> Snapshot saved at epoch {epoch} to '{snap_name}'")

    return history


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    train_ds = torchvision.datasets.FashionMNIST(
        './data', train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.FashionMNIST(
        './data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False, num_workers=0, pin_memory=True)

    excel_act =NDRActivation('./NDR.xlsx', col_x='x', col_y='y').to(device)
    model = ConfigurableMLP(activation=excel_act, w_bits=3, a_bits_in=8, a_bits_out=8).to(device)

    history = train_and_eval(model, train_loader, test_loader, device,
                            epochs=50, lr=1e-4, save_path='best_qat.pth')

    epochs = list(range(1, len(history['train_loss'])+1))
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, history['train_loss'], 'o-')
    plt.xlabel("Epoch"); plt.ylabel("Train Loss"); plt.title("Loss Curve")
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(epochs, history['test_acc'], 'o-')
    plt.xlabel("Epoch"); plt.ylabel("Test Accuracy (%)"); plt.title("Accuracy Curve")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    class_names = [str(i) for i in range(10)]
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

    with pd.ExcelWriter('qat_training_results.xlsx') as writer:
        df_hist = pd.DataFrame({
            'epoch': epochs,
            'train_loss': history['train_loss'],
            'test_acc': history['test_acc']
        })
        df_hist.to_excel(writer, sheet_name='history', index=False)
        df_cm.to_excel(writer, sheet_name='confusion_matrix')

    print("Training history and confusion matrix saved to 'qat_training_results.xlsx'")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()

