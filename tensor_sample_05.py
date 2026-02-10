import torch
import torch.nn as nn
import torch.optim as optim

#データの用意
x_raw = [[1.0], [2.0], [3.0], [4.0],  
         [6.0], [7.0], [8.0], [9.0]]  
# y: 合否 (0:不合格, 1:合格)
y_raw = [[0.0], [0.0], [0.0], [0.0],
         [1.0], [1.0], [1.0], [1.0]]

x_train = torch.tensor(x_raw)
y_train = torch.tensor(y_raw)
"""x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0],  

         [6.0], [7.0], [8.0], [9.0]])こういう書き方でもいいｐっぽい"""

#モデルの定義
model = nn.Sequential(   #nn.Sequential は PyTorch（torch.nn）におけるクラス。
    nn.Linear(1, 1),     ##これらもオブジェクト
    nn.Sigmoid()        #こいつも
)

optimizer = optim.SGD(model.parameters(), lr = 0.1)  #勾配とうか法のアルゴリズムを実行するためのインスタンス
criterion = nn.BCELoss()#criterionはlossを計算するための物ってこと？lossを計算するアルゴリズムを指定するための物ってこと

print("Learn start")

for i in range(2000):
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)

    loss.backward()
    optimizer.step()

    if i%200 == 0:
        print(f"{i}回目の学習 : loss = {loss.item():.4f}") 

print("学習完了")

test_student = torch.tensor([[5.0], [10.0],[6.0],[1.0]])
prediction = model(test_student)

for i in range(len(prediction)):
    h = test_student[i].item()
    prob = prediction[i].item()

    print(f"{h} 時間勉強した人の合格確率は {prob:.4f} です")


