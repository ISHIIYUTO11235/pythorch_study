import torch
import torch.optim as optim
import torch.nn as nn

#データ準備
x = torch.linspace(-5,5,100).unsqueeze(1)
y = x.pow(2)#このコードは、「x の中にあるすべての数字を、2乗（²）する」 という計算 pow は Power（パワー＝累乗）の略です。


# 個々からにゅ＾らるネットワーク
model = nn.Sequential(
    nn.Linear(1, 10),
    nn.ReLU(),#活性化関数。負の数字をゼロにする単純な関数
    nn.Linear(10, 1)

)



#オプティマイザ
optimizer = optim.Adam(model.parameters() , lr = 0.1)
criterion = nn.MSELoss()

#学習ループ
for i in range(2000):
    optimizer.zero_grad()       # 1. 勾配リセット
    y_pred = model(x)           # 2. 予測 (行列計算の塊)
    loss = criterion(y_pred, y) # 3. 誤差計算
    loss.backward()             # 4. 逆伝播
    optimizer.step()            # 5. 更新    基本的にこの流れ

    if i % 200 == 0:
        print(f"{i}回目の学習かんりょう：ロスは：{loss.item():.4f}です")

print("学習完了")

test_data = torch.tensor([3.0, -3.0])
test_data = test_data.unsqueeze(1)
prediction = model(test_data)

print("\n【実験結果】")
print(f"入力が  3.0 のとき、AIの答え: {prediction[0].item():.2f} (正解: 9.0)")
print(f"入力が -3.0 のとき、AIの答え: {prediction[1].item():.2f} (正解: 9.0)")