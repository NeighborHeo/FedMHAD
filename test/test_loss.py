import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import timm
from tqdm import tqdm

def setseed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

setseed(42)     

# GPU 사용 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 하이퍼파라미터 설정
batch_size = 64
learning_rate = 0.001
num_epochs = 5

# MNIST 데이터셋 다운로드 및 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='~/.data', train=True, download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 모델, 손실함수, 최적화 함수 정의
model = timm.create_model('resnet18', pretrained=False, num_classes=10, in_chans=1).to(device)
criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습 루프
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in tqdm(enumerate(train_loader)):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss1 = criterion1(outputs, labels)
        loss2 = criterion2(outputs, torch.nn.functional.one_hot(labels, num_classes=10).float().to(device))
        loss = 0.01*loss1 + 0.02*loss2
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    avg_loss = running_loss / len(train_loader.dataset)
    print("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, avg_loss))

# 마지막으로 모델의 손실값을 확인해보는 코드
test_dataset = datasets.MNIST(root='~/.data', train=False, download=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
test_loss = 0.0
acc = 0
with torch.no_grad():
    for inputs, labels in tqdm(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss1 = criterion1(outputs, labels)
        loss2 = criterion2(outputs, torch.nn.functional.one_hot(labels, num_classes=10).float().to(device))
        loss = 0.01*loss1 + 0.02*loss2
        test_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        acc += torch.sum(preds == labels.data)
        
test_loss /= len(test_loader.dataset)
print("Test Loss: {:.4f}".format(test_loss))
print("Test Accuracy: {:.2f}%".format(acc/len(test_loader.dataset)*100))