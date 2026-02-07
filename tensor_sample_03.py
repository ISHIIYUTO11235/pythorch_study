import torch
import torch.optim as optim
#1 データの準備
x_raw = [1.1, 1.3, 1.5, 2.0, 2.2, 2.9, 3.0, 3.2, 3.2, 3.7, 
         3.9, 4.0, 4.0, 4.1, 4.5, 4.9, 5.1, 5.3, 5.9, 6.0]
# y: 年収 (単位: ドル)
y_raw = [39343, 46205, 37731, 43525, 39891, 56642, 60150, 54445, 64445, 57189, 
         63218, 55794, 56957, 57081, 61111, 67938, 66029, 83088, 81363, 93940]

#でーた
x_train = torch.tensor(x_raw).unsqueeze(1)  #unsqueezeすることでただの一次元の配列を2次元ベクトルにしてあげる　そうすることでpythorchのAIモデルとの互換性を確保する。まあ細かいことなので呪文みたいなもの
y_train = torch.tensor(y_raw).unsqueeze(1)

#重要なこと、すうじがでかいとバグるらしいので正規化してあげる。大体一ケタかにケタくらいにした方がいいっぽい
x_scale = x_train / 10.0     # 1.1年 -> 0.11
y_scale = y_train / 10000.0  # 39343ドル -> 3.9343

#2　デモデルの定義
w = torch.randn(1,1, requires_grad = True)
b = torch.randn(1, requires_grad = True)


#3 学習の準備
optimizer = optim.SGD([w, b], lr=0.05)
print("--- 学習開始 ---")

for i in range(2000):
    optimizer.zero_grad()#初期化、PyTorchは計算した勾配をどんどん「足し算」していく癖があるので
#予測
    y_pred = x_scale @ w+b

    #誤差の計算（平均2乗誤差）
    loss = ((y_pred - y_scale)**2).mean() 
    '''(y_pred - y_scale): **「ズレ（誤差）」**を計算。

予測値と正解の差です。プラスになることもマイナスになることもあります。

**2: **「2乗」**します。

マイナスをプラスに変えるため（例: -5のズレも+25になる）。

大きな失敗をより厳しく評価するため。

.mean(): **「平均」**をとります。

これがご質問の核心です。データの個数（バッチサイズ）に関わらず、**「1データあたりの平均的なミスの大きさ」というたったひとつの数字（スカラ）**にまとめる役割です。'''
 
  
    loss.backward()#loss.backward(): パラメータ w, b はまだ書き換わりません。更新すべき量（勾配）を計算して、メモに残すだけです。
    optimizer.step()#optimizer.step(): ここで初めて、メモ（勾配）を見て w, b の数値を書き換えます。

    if i%200 == 0:
        print(f"{i}回目:Loss＝{loss.item():.4f}")

    
real_w = w.item() * (10000 / 10) 
real_b = b.item() * 10000  #正規化した奴を戻す

print(f"\n【AIが発見した給料の法則】")
print(f"年収 = {real_w:.0f}ドル × 経験年数 + {real_b:.0f}ドル")
print(f"(日本円で言うと: 1年ごとに約{real_w*150:.0f}円昇給、基本給{real_b*150:.0f}円)")

# 最後にテスト: 経験10年の人を予測させてみる
years = 10.0
prediction = real_w * years + real_b
print(f"\n【予測】経験{years}年の人の適正年収は...")
print(f"約 {prediction:.0f}ドル (約 {prediction*150/10000:.0f}万円) です！")