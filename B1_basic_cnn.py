import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

# ---------------------------------------------------------
# 1. ĐỊNH NGHĨA KIẾN TRÚC MÔ HÌNH (CƠ BẢN)  
# ---------------------------------------------------------
class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        # Khối Tích chập 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Khối Tích chập 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Khối Mạng nơ-ron truyền thẳng (Fully Connected)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, 10) # 10 classes

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8) # Flatten
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# ---------------------------------------------------------
# 2. HÀM CHÍNH ĐỂ CHẠY HUẤN LUYỆN
# ---------------------------------------------------------
def main():
    # Cấu hình thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")

    # Chuẩn bị dữ liệu
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False, num_workers=2)

    # Khởi tạo mô hình, Hàm mất mát và Tối ưu hóa
    model = BasicCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01) # Yêu cầu B1: Dùng SGD cơ bản

    # Vòng lặp huấn luyện
    epochs = 15
    print("-" * 40)
    print("Bắt đầu huấn luyện Bước 1 (CNN Cơ bản)")
    print("-" * 40)
    
    start_time = time.time()
    history = {'train_loss': [], 'val_acc': []}

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
        history['train_loss'].append(epoch_loss)
        
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
        history['val_acc'].append(epoch_acc)
        
        print(f"Epoch [{epoch+1:02d}/{epochs}] | Train Loss: {epoch_loss:.4f} | Val Accuracy: {epoch_acc:.2f}%")

    end_time = time.time()
    training_time = (end_time - start_time) / 60

    print("-" * 40)
    print("HOÀN THÀNH HUẤN LUYỆN BƯỚC 1")
    print(f"Tổng thời gian Train : {training_time:.2f} phút")
    print(f"Val Accuracy cao nhất: {max(history['val_acc']):.2f}%")

if __name__ == '__main__':
    main()
