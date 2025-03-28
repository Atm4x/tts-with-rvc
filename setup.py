from setuptools import setup, find_packages
import sys
import platform

install_requires = [
    # Основные библиотеки RVC и зависимостей
    "huggingface_hub",        # Для скачивания моделей
    "av",                     # Обработка аудио/видео (используется в audio.py)
    "nest_asyncio",           # Для запуска asyncio внутри другого event loop (используется в inference.py)
    "torch",                  # Основной фреймворк ML
    "edge-tts",               # TTS движок Microsoft Edge
    "numpy==1.26.0",          # Числовые операции
    "librosa==0.9.1",         # Анализ аудио
    "faiss-cpu==1.10.0",      # Поиск ближайших соседей для индекса
    "soundfile>=0.12.1",      # Чтение/запись аудио файлов
    "ffmpeg-python>=0.2.0",   # Обвязка для ffmpeg (используется в audio.py)
    "praat-parselmouth>=0.4.2",# Анализ речи (для pitch)
    "resampy>=0.4.2",         # Ресемплинг (зависимость librosa)
    "scikit-learn",           # Утилиты ML (зависимость librosa)
    "tqdm>=4.63.1",           # Прогресс-бары
    "audioread",              # Чтение аудио (зависимость librosa)
    "pyworld==0.3.2",         # Вокодер и анализ/синтез речи
    "torchcrepe==0.0.20",     # Алгоритм извлечения F0

    # Удаленные библиотеки (кандидаты)
    # "torchaudio",             # Не импортируется напрямую
    # "joblib>=1.1.0",          # Не импортируется
    # "omegaconf==2.3.0",       # Не импортируется
    # "hydra-core==1.3.2",      # Не импортируется
    # "antlr4-python3-runtime==4.9.3", # Не импортируется
    # "numba==0.59.0",          # Опциональная зависимость librosa, не импортируется напрямую
    # "llvmlite==0.42.0",       # Зависимость numba
    # "Cython",                 # Build-time зависимость
    # "pydub>=0.25.1",          # Не импортируется
    # "tensorboardX",           # Не импортируется
    # "json5",                  # Не импортируется
    # "matplotlib>=3.7.0",      # Не импортируется
    # "matplotlib-inline>=0.1.3", # Не импортируется
    # "Pillow>=9.1.1",          # Не импортируется
    # "tensorboard",            # Не импортируется
    # "uc-micro-py>=1.0.1",     # Не импортируется
    # "sympy>=1.11.1",          # Не импортируется
    # "tabulate>=0.8.10",       # Не импортируется
    # "PyYAML>=6.0",            # Не импортируется
    # "pyasn1>=0.4.8",          # Не импортируется
    # "pyasn1-modules>=0.2.8",  # Не импортируется
    # "fsspec>=2022.11.0",       # Не импортируется (иногда зависимость huggingface, но здесь вроде нет)
    # "absl-py>=1.2.0",         # Не импортируется
    # "colorama>=0.4.5",        # Не импортируется
    # "httpx>=0.23.0",          # Не импортируется (может быть зависимостью gradio/requests)
    # "fastapi==0.88",          # Не импортируется
    # "ffmpy==0.3.1"            # Не импортируется (используется ffmpeg-python)
]

python_version = f"{sys.version_info.major}{sys.version_info.minor}"
system = platform.system()

windows_fairseq_links = {
    "312": "https://github.com/Atm4x/fairseq-win-whl-3.12/releases/download/3.12/fairseq-0.12.3-cp312-cp312-win_amd64.whl",
    "311": "https://github.com/Atm4x/fairseq-win-whl-3.12/releases/download/3.12/fairseq-0.12.3-cp311-cp311-win_amd64.whl",
    "310": "https://github.com/Atm4x/fairseq-win-whl-3.12/releases/download/3.12/fairseq-0.12.3-cp310-cp310-win_amd64.whl",
}

linux_fairseq_links = {
    "312": "https://github.com/Atm4x/fairseq-win-whl-3.12/releases/download/3.12/fairseq-0.12.3-cp312-cp312-linux_x86_64.whl",
    "311": "https://github.com/Atm4x/fairseq-win-whl-3.12/releases/download/3.12/fairseq-0.12.3-cp311-cp311-linux_x86_64.whl",
    "310": "https://github.com/Atm4x/fairseq-win-whl-3.12/releases/download/3.12/fairseq-0.12.3-cp310-cp310-linux_x86_64.whl",
}

if system == "Windows" and python_version in windows_fairseq_links:
    install_requires.append(f"fairseq @ {windows_fairseq_links[python_version]}")
elif system == "Linux" and python_version in linux_fairseq_links:
    install_requires.append(f"fairseq @ {linux_fairseq_links[python_version]}")
else:
    install_requires.append("fairseq @ git+https://github.com/One-sixth/fairseq.git")

setup(
    name='tts_with_rvc',
    version='0.1.6',
    description='TTS with RVC pipeline',
    author='Atm4x',
    packages=find_packages(),
    install_requires=install_requires,
    scripts=['tts_with_rvc/inference.py']
)