import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel

# =============================================================================
# BiGRU — двунаправленная GRU
# =============================================================================

class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        # Возвращаем только первый элемент (выходные представления),
        # т. к. (h_n, c_n) не используется в исходной логике.
        return self.gru(x)[0]


# =============================================================================
# РResidual-блок на свёртках
# =============================================================================

class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.01):
        super(ConvBlockRes, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))
            self.is_shortcut = True
        else:
            self.is_shortcut = False

    def forward(self, x):
        if self.is_shortcut:
            return self.conv(x) + self.shortcut(x)
        else:
            return self.conv(x) + x


# =============================================================================
# Кодер (Encoder) — выполняет постепенное уменьшение разрешения (downsampling)
# =============================================================================

class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        in_size,
        n_encoders,
        kernel_size,
        n_blocks,
        out_channels=16,
        momentum=0.01,
    ):
        super(Encoder, self).__init__()
        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        self.latent_channels = []
        for i in range(self.n_encoders):
            self.layers.append(
                ResEncoderBlock(
                    in_channels, out_channels, kernel_size, n_blocks, momentum=momentum
                )
            )
            self.latent_channels.append([out_channels, in_size])
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x):
        concat_tensors = []
        x = self.bn(x)
        for i in range(self.n_encoders):
            # "_" содержит результат без pooling,
            # "x" — результат после pooling, используемый для следующего слоя.
            _, x = self.layers[i](x)
            concat_tensors.append(_)
        return x, concat_tensors


# =============================================================================
# Блок кодировщика с резидуальными свёртками и опциональным pooling
# =============================================================================

class ResEncoderBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01
    ):
        super(ResEncoderBlock, self).__init__()
        self.n_blocks = n_blocks
        self.conv = nn.ModuleList()
        self.conv.append(ConvBlockRes(in_channels, out_channels, momentum))
        for i in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if self.kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for i in range(self.n_blocks):
            x = self.conv[i](x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        else:
            # Если нет pooling, то возвращаем x в обе переменные (логика не меняется).
            return x


# =============================================================================
# Промежуточные слои (Intermediate) — цепочка блоков без downsampling
# =============================================================================

class Intermediate(nn.Module):
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super(Intermediate, self).__init__()
        self.n_inters = n_inters
        self.layers = nn.ModuleList()
        self.layers.append(
            ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum)
        )
        for i in range(self.n_inters - 1):
            self.layers.append(
                ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum)
            )

    def forward(self, x):
        for i in range(self.n_inters):
            x = self.layers[i](x)
        return x


# =============================================================================
# Блок декодера с транспонированными свёртками и сверточными Residual-блоками
# =============================================================================

class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        super(ResDecoderBlock, self).__init__()
        # Вычисляем output_padding для транспонированной свёртки,
        # чтобы логика upsampling совпадала с downsampling.
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.n_blocks = n_blocks
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                output_padding=out_padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList()
        # Здесь объединяем (cat) с соответствующим "concat_tensor" из кодера,
        # поэтому in_channels = out_channels*2 при первом блоке conv2.
        self.conv2.append(ConvBlockRes(out_channels * 2, out_channels, momentum))
        for i in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x, concat_tensor):
        x = self.conv1(x)
        x = torch.cat((x, concat_tensor), dim=1)
        for i in range(self.n_blocks):
            x = self.conv2[i](x)
        return x


# =============================================================================
# Декодер (Decoder) — возвращает пространственное разрешение к исходному
# =============================================================================

class Decoder(nn.Module):
    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        self.n_decoders = n_decoders
        for i in range(self.n_decoders):
            out_channels = in_channels // 2
            self.layers.append(
                ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum)
            )
            in_channels = out_channels

    def forward(self, x, concat_tensors):
        # Обратите внимание, что concat_tensors идёт в обратном порядке.
        for i in range(self.n_decoders):
            x = self.layers[i](x, concat_tensors[-1 - i])
        return x


# =============================================================================
# Глубокая U-образная сеть (DeepUnet)
# =============================================================================

class DeepUnet(nn.Module):
    def __init__(
        self,
        kernel_size,
        n_blocks,
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16,
    ):
        super(DeepUnet, self).__init__()
        self.encoder = Encoder(
            in_channels, 128, en_de_layers, kernel_size, n_blocks, en_out_channels
        )
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2,
            self.encoder.out_channel,
            inter_layers,
            n_blocks,
        )
        self.decoder = Decoder(
            self.encoder.out_channel, en_de_layers, kernel_size, n_blocks
        )

    def forward(self, x):
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x


# =============================================================================
# Модель End-to-End (E2E), объединяющая DeepUnet и BiGRU/FC для классификации
# =============================================================================

class E2E(nn.Module):
    def __init__(
        self,
        n_blocks,
        n_gru,
        kernel_size,
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16,
    ):
        super(E2E, self).__init__()
        self.unet = DeepUnet(
            kernel_size,
            n_blocks,
            en_de_layers,
            inter_layers,
            in_channels,
            en_out_channels,
        )
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))

        # В зависимости от наличия BiGRU формируем разные ветки.
        if n_gru:
            self.fc = nn.Sequential(
                BiGRU(3 * 128, 256, n_gru),
                nn.Linear(512, 360),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )
        else:
            # Здесь предполагается, что nn.N_MELS и nn.N_CLASS доступны глобально
            # (либо в окружении, либо где-то определены).
            self.fc = nn.Sequential(
                nn.Linear(3 * nn.N_MELS, nn.N_CLASS),
                nn.Dropout(0.25),
                nn.Sigmoid()
            )

    def forward(self, mel):
        # Подготовка к входу в U-Net: транспонируем B x (T x freqs) → B x 1 x freqs x T
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


# =============================================================================
# Модуль для вычисления мел-спектрограммы (MelSpectrogram) с кэшированием окон
# =============================================================================

class MelSpectrogram(torch.nn.Module):
    def __init__(
        self,
        is_half,
        n_mel_channels,
        sampling_rate,
        win_length,
        hop_length,
        n_fft=None,
        mel_fmin=0,
        mel_fmax=None,
        clamp=1e-5,
    ):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        self.hann_window = {}
        mel_basis = mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
            htk=True,
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.n_fft = win_length if n_fft is None else n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp
        self.is_half = is_half

    def forward(self, audio, keyshift=0, speed=1, center=True):
        # Подгоняем размеры n_fft, win_length, hop_length при сдвиге по тону (keyshift) и скорости (speed).
        factor = 2 ** (keyshift / 12)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_length_new = int(np.round(self.win_length * factor))
        hop_length_new = int(np.round(self.hop_length * speed))

        # Ключ для кэширования окон Хэнна (учёт сдвига и устройства).
        keyshift_key = str(keyshift) + "_" + str(audio.device)
        if keyshift_key not in self.hann_window:
            self.hann_window[keyshift_key] = torch.hann_window(win_length_new).to(
                audio.device
            )
        # Вычисление STFT:
        fft = torch.stft(
            audio,
            n_fft=n_fft_new,
            hop_length=hop_length_new,
            win_length=win_length_new,
            window=self.hann_window[keyshift_key],
            center=center,
            return_complex=True,
        )
        magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))

        # Корректировка размера при keyshift, чтобы впоследствии правильно умножать на mel_basis.
        if keyshift != 0:
            size = self.n_fft // 2 + 1
            resize = magnitude.size(1)
            if resize < size:
                magnitude = F.pad(magnitude, (0, 0, 0, size - resize))
            magnitude = magnitude[:, :size, :] * self.win_length / win_length_new

        # Умножение на матрицу мел-базиса.
        mel_output = torch.matmul(self.mel_basis, magnitude)
        if self.is_half:
            mel_output = mel_output.half()

        # Логарифм с clamp для избежания Nan-значений при малых амплитудах.
        log_mel_spec = torch.log(torch.clamp(mel_output, min=self.clamp))
        return log_mel_spec


# Глобальная переменная для хранения загруженного state_dict (чекпойнта).
ckpt = None


# =============================================================================
# Класс RMVPE — основная логика предсказания F0
# =============================================================================

class RMVPE:
    def __init__(self, model_path, is_half, device=None):
        self.resample_kernel = {}
        global ckpt

        # Создаём модель E2E.
        model = E2E(4, 1, (2, 2))

        # Если чекпойнт ещё не был загружен, грузим и сохраняем в ckpt (кэш).
        if ckpt is None:
            ckpt = torch.load(model_path, map_location="cpu")

        # Загружаем веса в модель.
        model.load_state_dict(ckpt)
        model.eval()

        if is_half:
            model = model.half()
        self.model = model
        self.resample_kernel = {}
        self.is_half = is_half

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Экземпляр MelSpectrogram для извлечения мел-спектрограм.
        self.mel_extractor = MelSpectrogram(
            is_half, 128, 16000, 1024, 160, None, 30, 8000
        ).to(device)

        # Перенос модели на нужное устройство.
        self.model = self.model.to(device)

        # Предварительно вычисляем соответствие индексов (360) и центов (cents_mapping).
        cents_mapping = 20 * np.arange(360) + 1997.3794084376191
        # Пэддим на 4 с каждой стороны, чтобы захватывать окрестность при локальном поиске.
        self.cents_mapping = np.pad(cents_mapping, (4, 4))  # 368

    def mel2hidden(self, mel):
        with torch.no_grad():
            n_frames = mel.shape[-1]
            # Пэддим до ближайшего числа, кратного 32, для удобства работы в conv/GRU.
            mel = F.pad(
                mel, (0, 32 * ((n_frames - 1) // 32 + 1) - n_frames), mode="reflect"
            )
            hidden = self.model(mel)
            return hidden[:, :n_frames]

    def decode(self, hidden, thred=0.03):
        cents_pred = self.to_local_average_cents(hidden, thred=thred)
        # Преобразуем центы в частоту F0.
        f0 = 10 * (2 ** (cents_pred / 1200))
        # Там, где cents_pred=0 (ниже threshold), ставим F0=0.
        f0[f0 == 10] = 0
        return f0

    def infer_from_audio(self, audio, thred=0.03):
        # Преобразуем numpy → torch.tensor, переносим на нужное устройство.
        audio = torch.from_numpy(audio).float().to(self.device).unsqueeze(0)
        # Вычисляем мел-спектрограмму.
        mel = self.mel_extractor(audio, center=True)
        # Прогоняем через модель, получаем скрытые представления.
        hidden = self.mel2hidden(mel)
        hidden = hidden.squeeze(0).cpu().numpy()
        if self.is_half:
            hidden = hidden.astype("float32")
        # Декодируем скрытые представления в F0.
        f0 = self.decode(hidden, thred=thred)
        return f0

    def to_local_average_cents(self, salience, thred=0.05):
        # Находим максимальные индексы (центры) по каждому фрейму.
        center = np.argmax(salience, axis=1)

        # Пэддим массив salience слева и справа на 4, чтобы не выходить за границы при локальном усреднении.
        salience = np.pad(salience, ((0, 0), (4, 4)))
        center += 4

        # Готовим «окна» шириной 9 (center-4 : center+5).
        starts = center - 4
        ends = center + 5

        todo_salience = []
        todo_cents_mapping = []
        for idx in range(salience.shape[0]):
            todo_salience.append(salience[:, starts[idx] : ends[idx]][idx])
            todo_cents_mapping.append(self.cents_mapping[starts[idx] : ends[idx]])

        todo_salience = np.array(todo_salience)
        todo_cents_mapping = np.array(todo_cents_mapping)

        # Усреднение центов взвешенное по матрице salience в окрестности пика.
        product_sum = np.sum(todo_salience * todo_cents_mapping, 1)
        weight_sum = np.sum(todo_salience, 1)
        devided = product_sum / weight_sum

        # Участки, где максимум salience ниже thred, считаем беззвучными (0).
        maxx = np.max(salience, axis=1)
        devided[maxx <= thred] = 0

        return devided