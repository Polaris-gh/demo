# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pywt
from statsmodels.graphics.tsaplots import plot_acf
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
from PIL import Image

matplotlib.rc("font", family='Microsoft YaHei')

file_path = "WHTC.xlsx"
df = pd.read_excel(file_path)

nox_signal = df["NOx排放浓度"].values
nox_len = len(nox_signal)
T = 1.0

# 傅里叶变换
yf = fft(nox_signal - np.mean(nox_signal))
xf = fftfreq(nox_len, T)[:nox_len // 2]

plt.figure(figsize=(10, 5))
plt.plot(xf, 2.0 / nox_len * np.abs(yf[:nox_len // 2]))
plt.title("NOx排放信号的频域表示（FFT）")
plt.xlabel("频率 (Hz)")
plt.ylabel("幅值")
plt.grid(True)
plt.ylim(0, 200)
plt.tight_layout()
plt.savefig("FFT.png")
plt.show()
img = Image.open("FFT.png").convert("L")
img.save("FFT_gray.png")

target_column = "NOx排放浓度"
df_denoising = pd.DataFrame()


def butter_lowpass_filter(data, cutoff, fs, order=1):
    nyq = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = cutoff / nyq
    [b, a] = butter(order, normal_cutoff, btype='low')
    y = filtfilt(b, a, data)
    return y


def estimate_cutoff(signal, fs=1.0, energy_threshold=0.95):
    N = len(signal)
    yf = np.abs(fft(signal))[:N // 2]
    xf = fftfreq(N, 1 / fs)[:N // 2]
    yf[0] = 0

    energy = np.cumsum(yf ** 2)
    energy /= energy[-1]
    cutoff_index = np.searchsorted(energy, energy_threshold)
    return xf[cutoff_index]


for col in df.columns:
    raw_signal = df[col].values
    cutoff = estimate_cutoff(raw_signal, fs=1.0, energy_threshold=0.95)
    filtered = butter_lowpass_filter(raw_signal, cutoff=cutoff, fs=1.0)
    df_denoising[col] = filtered[:len(df)]
# 皮尔逊相关系数
denoised_target_col = f"{target_column}"
corr_matrix = df_denoising.corr(method="pearson")
target_corr = corr_matrix[denoised_target_col].dropna()

# 获取最相关的前 N 个变量
n_multiple = 10
top_features = (
    target_corr.drop(labels=[denoised_target_col])
    .abs()
    .sort_values(ascending=False)
    .head(n_multiple)
    .index.tolist()
)

final_columns = [denoised_target_col] + top_features
df_filtered = df_denoising[final_columns].copy()

# 热力图
plt.figure(figsize=(6, 6))
plt.gray()
sns.heatmap(df_filtered.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title(f"{target_column}相关性热力图")
plt.tight_layout()
plt.savefig("multiple_Pearson.png")
plt.show()
img = Image.open("multiple_Pearson.png").convert("L")
img.save("multiple_Pearson_gray.png")

# ACF自相关系数
nox_series = df_denoising[denoised_target_col]
plt.figure(figsize=(8, 8))
plot_acf(nox_series, lags=10)
plt.gray()
plt.ylim(0, 1.2)
plt.title("NOx排放的自相关分析")
plt.xlabel("滞后阶数")
plt.ylabel("相关系数")
plt.savefig("single_ACF.png")
plt.show()
img = Image.open("single_ACF.png").convert("L")
img.save("single_ACF_gray.png")

# 去噪前后对比图
plt.figure(figsize=(15, 10))
plt.subplot(3, 1, 1)
plt.plot(df[target_column], label='去噪前', color='blue')
plt.title('NOx排放浓度（去噪前）')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(df_denoising[f"{target_column}"], label='去噪后', color='red')
plt.title('NOx排放浓度（去噪后）')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(df[target_column], label='去噪前', color='red', linewidth=2)
plt.plot(df_denoising[f"{target_column}"], label='去噪后', color='blue', alpha=0.7)
plt.title(f'{target_column} 去噪前后对比图')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

nox_signal = nox_series.values
NOx_len = len(nox_signal)
T = 1.0

# 傅里叶变换
yf = fft(nox_signal - np.mean(nox_signal))
xf = fftfreq(NOx_len, T)[:NOx_len // 2]

plt.figure(figsize=(10, 5))
plt.gray()
plt.plot(xf, 2.0 / NOx_len * np.abs(yf[:NOx_len // 2]))
plt.title("NOx排放信号的频域表示（FFT）")
plt.xlabel("频率 (Hz)")
plt.ylabel("幅值")
plt.ylim(0, 200)
plt.grid(True)
plt.tight_layout()
plt.savefig("FFT_de.png")
plt.show()
img = Image.open("FFT_de.png").convert("L")
img.save("FFT_de_gray.png")


def smooth_std(signal):
    return np.std(np.diff(signal))


for col in df.columns:
    original = df[col].values
    denoised = df_denoising[f"{col}"].values
    print(f"{col} 原始信号一阶差分 std : {smooth_std(original):.4f}")
    print(f"{col} 去噪后一阶差分 std  : {smooth_std(denoised):.4f}")
    print("")

# ==== Step 8: 保存去噪数据 ====
df_filtered.to_excel("Denoising_WHTC.xlsx", index=False)

# %%
import pandas as pd
import torch.utils.data as Data
from sklearn.preprocessing import StandardScaler
from joblib import dump
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# 读取数据
original_data = pd.read_excel("Denoising_WHTC.xlsx")

# 目标数据
aim_data = original_data[['NOx排放浓度']]
# 相关数据，用于预测
target_column = aim_data.columns
input_features = original_data.columns.drop(target_column)
all_data = original_data[input_features].select_dtypes(include=['number'])
print(all_data.columns.tolist())
# 归一化处理
scaler_input = StandardScaler()
all_data = scaler_input.fit_transform(all_data)
dump(scaler_input, "scaler_input")

scaler_target = StandardScaler()
NOx_data = scaler_target.fit_transform(aim_data)
dump(scaler_target, "scaler_target")


# %%
def split_data(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    train_set, temp_set = train_test_split(all_data, train_size=train_ratio, shuffle=False)
    val_set, test_set = train_test_split(temp_set, train_size=val_ratio / (val_ratio + test_ratio), shuffle=False)

    train_label = NOx_data[:len(train_set)]
    val_label = NOx_data[len(train_set):len(train_set) + len(val_set)]
    test_label = NOx_data[len(train_set) + len(val_set):]

    dump(train_set, 'train_set')
    dump(val_set, 'val_set')
    dump(test_set, 'test_set')
    dump(train_label, 'train_label')
    dump(val_label, 'val_label')
    dump(test_label, 'test_label')

    plt.figure(figsize=(15, 10))

    # 绘制训练集标签
    plt.subplot(3, 1, 1)
    plt.plot(train_label, label='Train Label', color='blue')
    plt.title('Train Label')
    plt.legend()

    # 绘制验证集标签
    plt.subplot(3, 1, 2)
    plt.plot(val_label, label='Validation Label', color='green')
    plt.title('Validation Label')
    plt.legend()

    # 绘制测试集标签
    plt.subplot(3, 1, 3)
    plt.plot(test_label, label='Test Label', color='red')
    plt.title('Test Label')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return train_set, val_set, test_set, train_label, val_label, test_label


def data_window_maker(x_var, y_label, window_size, win_stride=1):
    multiple_x, single_x, data_y = [], [], []
    for i in range(0, len(x_var) - window_size, win_stride):
        multiple_x_window = x_var[i:i + window_size, :]
        single_x_window = y_label[i:i + window_size]
        target = y_label[i + window_size]
        multiple_x.append(multiple_x_window)
        single_x.append(single_x_window)
        data_y.append(target)
    return (torch.tensor(np.array(multiple_x)).float(),
            torch.tensor(np.array(single_x)).float(),
            torch.tensor(np.array(data_y)).float())


def dataloader(batch_size, window_size, win_stride, workers=2):
    train_set = load('train_set')
    train_label = load('train_label')
    val_set = load('val_set')
    val_label = load('val_label')
    test_set = load('test_set')
    test_label = load('test_label')

    train_multiple, train_single, train_label = data_window_maker(train_set, train_label, window_size, win_stride)

    val_multiple, val_single, val_label = data_window_maker(val_set, val_label, window_size, win_stride)

    test_multiple, test_single, test_label = data_window_maker(test_set, test_label, window_size, win_stride)

    train_loader = Data.DataLoader(dataset=Data.TensorDataset(train_multiple, train_single, train_label),
                                   batch_size=batch_size, num_workers=workers, drop_last=True)
    val_loader = Data.DataLoader(dataset=Data.TensorDataset(val_multiple, val_single, val_label),
                                 batch_size=batch_size, num_workers=workers, drop_last=True)
    test_loader = Data.DataLoader(dataset=Data.TensorDataset(test_multiple, test_single, test_label),
                                  batch_size=batch_size, num_workers=workers, drop_last=True)

    # 检查训练数据形状
    for x1, x2, label in train_loader:
        print(f"Train Multi Input shape: {x1.shape}")
        print(f"Train NOx Input shape  : {x2.shape}")
        print(f"Train Label shape      : {label.shape}")
        break
    return train_loader, val_loader, test_loader


# %%
import torch
import torch.nn as nn


class SimpleAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, h_n, lstm_outputs):
        weights = torch.softmax(self.attn(lstm_outputs).squeeze(-1), dim=1)
        context = torch.sum(weights.unsqueeze(-1) * lstm_outputs, dim=1)
        return context


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(1)
        repeated_hidden = hidden.unsqueeze(1).repeat(1, max_len, 1)
        energy = torch.tanh(self.attn(torch.cat((repeated_hidden, encoder_outputs), dim=2)))
        attention_scores = self.v(energy).squeeze(2)
        attention_weights = nn.functional.softmax(attention_scores, dim=1)
        context_vector = (encoder_outputs * attention_weights.unsqueeze(2)).sum(dim=1)
        return context_vector


class CNNLSTMAttention(nn.Module):
    def __init__(self, input_dim, conv_archs, hidden_layer_sizes, attention_dim, output_dim=None):
        super().__init__()
        self.output_dim = output_dim
        self.conv_arch = conv_archs
        self.input_channels = input_dim
        self.cnn_features = self.make_layers()

        self.num_layers = len(hidden_layer_sizes)
        self.lstm_layers = nn.ModuleList()
        self.lstm_layers.append(nn.LSTM(conv_archs[-1][-1], hidden_layer_sizes[0], batch_first=True))
        for i in range(1, self.num_layers):
            self.lstm_layers.append(nn.LSTM(hidden_layer_sizes[i - 1], hidden_layer_sizes[i], batch_first=True))

        self.attention = Attention(attention_dim)
        if output_dim is not None:
            self.linear = nn.Linear(hidden_layer_sizes[-1], output_dim)

    def make_layers(self):
        layers = []
        for (num_convs, out_channels) in self.conv_arch:
            for _ in range(num_convs):
                layers.append(nn.Conv1d(self.input_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                self.input_channels = out_channels
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, input_seq):
        input_seq = input_seq.permute(0, 2, 1)
        cnn_features = self.cnn_features(input_seq)
        lstm_out = cnn_features.permute(0, 2, 1)
        hidden_states = []
        for lstm in self.lstm_layers:
            lstm_out, hidden = lstm(lstm_out)
            hidden_states += hidden
        attention_features = self.attention(hidden_states[-1][-1], lstm_out)
        if hasattr(self, 'linear'):
            return self.linear(attention_features)
        else:
            return attention_features


class DualChannelModel(nn.Module):
    def __init__(self, input_dim_multi, conv_archs, hidden_sizes_multi, hidden_size_single, attention_dim, output_dim):
        super().__init__()
        self.cnn_lstm = CNNLSTMAttention(
            input_dim=input_dim_multi,
            conv_archs=conv_archs,
            hidden_layer_sizes=hidden_sizes_multi,
            attention_dim=attention_dim,
            output_dim=None
        )

        # 自回归通道：使用单层 LSTM
        self.lstm2 = nn.LSTM(1, hidden_size_single, batch_first=True)
        self.simple_attn = SimpleAttention(hidden_size_single)

        # 融合后输入 FC 层的维度
        fusion_dim = hidden_sizes_multi[-1] + hidden_size_single
        self.fc = nn.Linear(fusion_dim, output_dim)

    def forward(self, x_multi, x_nox):
        out1 = self.cnn_lstm(x_multi)  # 多变量通道
        lstm_out2, (h2, _) = self.lstm2(x_nox)  # 自回归通道
        out2 = self.simple_attn(h2[-1], lstm_out2)
        return self.fc(torch.cat([out1, out2], dim=1))


def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    for item in params:
        print(f'{item:>6}')
    print(f'______\n{sum(params):>6}')


# %%
# 固定随机种子及设备设置
torch.manual_seed(100)
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
# input_dim = n_multiple # 输入维度
conv_archs = ((1, 16), (1, 32))  # CNN 卷积池化结构
hidden_sizes_multi = [32, 64]
hidden_size_single = 32
attention_dim = hidden_sizes_multi[-1]  # 注意力层维度
output_dim = 1  # 输出维度
learn_rate = 0.003  # 学习率
loss_function = nn.MSELoss()


# %%
class EarlyStopping:
    def __init__(self, patience, delta):
        self.patience = int(patience)
        self.delta = float(delta)
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# %%

# %%
import matplotlib
from joblib import load
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np


def model_train_dual(epochs, loss_function, batch_size, device, window_size, win_stride):
    torch.manual_seed(100)
    start_time = time.time()
    print(f'\nTraining with window size: {window_size}')

    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.70)
    early_stopping = EarlyStopping(patience=8, delta=0.001)

    # 加载双通道数据
    train_loader, val_loader, test_loader = dataloader(batch_size, window_size, win_stride)

    train_mse_list = []
    val_mse_list = []

    for epoch in range(epochs):
        model.train()
        train_mse = []
        for x1, x2, labels in train_loader:
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
            optimizer.zero_grad()
            y_pred = model(x1, x2)
            loss = loss_function(y_pred, labels)
            train_mse.append(loss.item())
            loss.backward()
            optimizer.step()

        train_avg_mse = np.average(train_mse)
        train_mse_list.append(train_avg_mse)
        print(f'Epoch {epoch + 1:2d} Train MSE: {train_avg_mse:10.8f}')

        with torch.no_grad():
            model.eval()
            val_mse = []
            for x1, x2, label in val_loader:
                x1, x2, label = x1.to(device), x2.to(device), label.to(device)
                val_pred = model(x1, x2)
                val_loss = loss_function(val_pred, label)
                val_mse.append(val_loss.item())

            val_avg_mse = np.average(val_mse)
            val_mse_list.append(val_avg_mse)
            print(f'Epoch {epoch + 1:2d} Val MSE: {val_avg_mse:10.8f}')

        scheduler.step(val_avg_mse)
        early_stopping(val_avg_mse)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    torch.save(model.state_dict(), 'best_model_dual_channel.pt')
    print(f'\nTraining finished in {time.time() - start_time:.0f} seconds')

    # 绘图
    matplotlib.rc("font", family='Microsoft YaHei')
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_mse_list) + 1), train_mse_list, label='Train MSE', color='blue')
    plt.plot(range(1, len(val_mse_list) + 1), val_mse_list, label='Val MSE', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.title(f'Training & Validation MSE (window={window_size})')
    plt.grid(True)
    plt.show()

    return model


# %%
if __name__ == '__main__':
    # 划分数据集
    train_set, val_set, test_set, train_label, val_label, test_label = split_data()

    # 超参数设置
    epochs = 100
    window_size = 5
    win_stride = 1
    batch_size = 32
    input_dim_multi = n_multiple
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")

    # 打印模型结构
    model = DualChannelModel(
        input_dim_multi=n_multiple,
        conv_archs=conv_archs,
        hidden_sizes_multi=hidden_sizes_multi,
        hidden_size_single=hidden_size_single,
        attention_dim=attention_dim,
        output_dim=output_dim
    ).to(device)

    count_parameters(model)
    print(model)

    # 模型训练
    model = model_train_dual(epochs, loss_function, batch_size, device, window_size, win_stride)

    # %%
    model.load_state_dict(torch.load('best_model_dual_channel.pt', weights_only=False))
    model = model.to(device)

    # 获取测试集
    _, _, test_loader = dataloader(batch_size, window_size, win_stride)

    original_values = []
    predicted_values = []

    model.eval()
    with torch.no_grad():
        for x1, x2, label in test_loader:
            x1, x2, label = x1.to(device), x2.to(device), label.to(device)
            pred = model(x1, x2)
            predicted_values.extend(pred.cpu().numpy())
            original_values.extend(label.cpu().numpy())

    # 反归一化
    original_values = np.array(original_values)
    predicted_values = np.array(predicted_values)
    NOx_scaler = load('scaler_target')
    original_values = NOx_scaler.inverse_transform(original_values)
    predicted_values = NOx_scaler.inverse_transform(predicted_values)

    # 评估指标
    r2 = r2_score(original_values, predicted_values)
    mse = mean_squared_error(original_values, predicted_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(original_values, predicted_values)

    # 打印结果
    print('*' * 50)
    print(f'R^2: {r2:.4f}')
    print(f'MSE : {mse:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'MAE : {mae:.4f}')
    print('*' * 50)

    # 可视化
    plt.figure(figsize=(12, 6))
    plt.plot(original_values, label='真实值', color='hotpink')
    plt.plot(predicted_values, label='预测值', color='cyan')
    plt.legend()
    plt.title('NOx 浓度预测结果')
    plt.savefig('NOx_Output_DualChannel.png')
    plt.show()

# %%
import torch
import torch.onnx

batch_size = 1
seq_len = 5
feature_dim = 10

# 构建模型
model = DualChannelModel(
    input_dim_multi=n_multiple,
    conv_archs=conv_archs,
    hidden_sizes_multi=hidden_sizes_multi,
    hidden_size_single=hidden_size_single,
    attention_dim=attention_dim,
    output_dim=output_dim
)
model.eval()

# 虚拟输入
x_multi = torch.randn(batch_size, seq_len, feature_dim)
x_nox = torch.randn(batch_size, seq_len, 1)

# 不显式传 h0/c0
example_inputs = (x_multi, x_nox)

# 导出 ONNX
torch.onnx.export(
    model,
    example_inputs,
    "dual_channel_model.onnx",
    input_names=["x_multi", "x_nox"],
    output_names=["output"],
    dynamic_axes={
        "x_multi": {0: "batch_size"},
        "x_nox": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    opset_version=11
)

print("保存为dual_channel_model.onnx")
