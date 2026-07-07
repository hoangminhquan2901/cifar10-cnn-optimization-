
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

# ---------------------------------------------------------
# 1. KIẾN TRÚC MÔ HÌNH CẢI TIẾN
# ---------------------------------------------------------
class ImprovedCNN(nn.Module):
    def __init__(self, dropout_rate):
        super(ImprovedCNN, self).__init__()
        
        # Khối Tích chập 1 + Batch Normalization + Dropout
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) # Thêm Batch Normalization
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(dropout_rate) # Thêm Dropout

        # Khối Tích chập 2 + Batch Normalization + Dropout
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) # Thêm Batch Normalization
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(dropout_rate) # Thêm Dropout

        # Khối Mạng nơ-ron truyền thẳng (Fully Connected)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate) # Dropout cho lớp tuyến tính
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.dropout1(self.pool1(self.relu1(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(self.relu2(self.bn2(self.conv2(x)))))
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout3(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x

# ---------------------------------------------------------
# 2. HÀM HUẤN LUYỆN
# ---------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False, num_workers=2)

    # =========================================================
    DROPOUT_RATE = 0.5
    OPTIMIZER_TYPE = 'Adam'
    # =========================================================

    # Khởi tạo mô hình với Dropout Rate truyền vào
    model = ImprovedCNN(dropout_rate=DROPOUT_RATE).to(device)
    criterion = nn.CrossEntropyLoss()

    # Lựa chọn Optimizer
    if OPTIMIZER_TYPE == 'SGD_Momentum':
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif OPTIMIZER_TYPE == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("-" * 50)
    print(f"BẮT ĐẦU TRAIN: Cấu hình Dropout = {DROPOUT_RATE}, Optimizer = {OPTIMIZER_TYPE}")
    print("-" * 50)

    epochs = 15
    start_time = time.time()
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Train
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(trainloader)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        epoch_acc = 100 * correct / total
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            
        print(f"Epoch [{epoch+1:02d}/{epochs}] | Train Loss: {epoch_loss:.4f} | Val Accuracy: {epoch_acc:.2f}%")

    end_time = time.time()
    print("-" * 50)
    print(f"TỔNG KẾT CẤU HÌNH: {OPTIMIZER_TYPE} + Dropout {DROPOUT_RATE}")
    print(f"Tổng thời gian Train : {(end_time - start_time) / 60:.2f} phút")
    print(f"Val Accuracy cao nhất: {best_acc:.2f}%")
    print("-" * 50)

if __name__ == '__main__':
    main()
