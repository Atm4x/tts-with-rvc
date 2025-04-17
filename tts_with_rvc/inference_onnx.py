# File: tts_with_rvc/inference_onnx.py
# Content:
import os
import edge_tts as tts
import asyncio
import hashlib
from datetime import datetime
import nest_asyncio
import threading
from huggingface_hub import hf_hub_download
import onnxruntime
import soundfile
import logging
import warnings
import tempfile  

import concurrent

logger = logging.getLogger(__name__)
logging.getLogger('onnxruntime').setLevel(logging.ERROR)

nest_asyncio.apply()

class TTS_RVC:
    """
    Combines Edge TTS for text-to-speech with RVC (ONNX) for voice conversion.

    Args:
        model_path (str): Path to the RVC .onnx model file.
        voice (str, optional): Edge TTS voice identifier (e.g., "ru-RU-DmitryNeural"). Defaults to "ru-RU-DmitryNeural".
        index_path (str, optional): Path to the RVC .index file. Defaults to "".
        output_directory (str, optional): Directory to save voiceovered audios. Defaults to 'temp'.
        f0_method (str, optional): F0 extraction method ('pm', 'dio', 'harvest', 'crepe', 'rmvpe'). Defaults to "pm".
        sampling_rate (int, optional): Target sampling rate for RVC model. Defaults to 40000.
        hop_size (int, optional): Hop size for RVC model. Defaults to 512.
        device (str, optional): ONNX Runtime execution provider ('dml', 'cuda:0', 'cpu', etc.). Defaults to "dml".
        tmp_directory (str, optional): Directory for temporary TTS files. Defaults to system temp dir.
    """
    def __init__(self, model_path, tmp_directory=None, voice="ru-RU-DmitryNeural", index_path="",
                 f0_method="pm", output_directory=None, sampling_rate=40000,
                 hop_size=512, device="dml", input_directory=None):

        if input_directory is not None:
            warnings.warn("Parameter 'input_directory' is deprecated and will be deleted in the future. "
                          "Use tmp_directory instead.", DeprecationWarning, stacklevel=2)
            if tmp_directory is None:
                self.tmp_directory = input_directory
        else:
            self.tmp_directory = tmp_directory

        # Устанавливаем временную директорию, если не задана
        if self.tmp_directory is None:
            self.tmp_directory = os.path.join(tempfile.gettempdir(), "tts_with_rvc_onnx")
            os.makedirs(self.tmp_directory, exist_ok=True)
        elif not os.path.exists(self.tmp_directory):
             os.makedirs(self.tmp_directory, exist_ok=True)

        self.current_voice = voice
        self.can_speak = True
        self.current_model = model_path
        self.output_directory = output_directory
        self.f0_method = f0_method
        self.sampling_rate = sampling_rate
        self.hop_size = hop_size
        self.device = device
        self.onnx_model = None # Инициализируем как None

        # Создаем выделенный цикл событий для асинхронных операций
        self.loop = asyncio.new_event_loop()
        self._loop_thread = None
        self._start_background_loop()

        if index_path != "":
            if not os.path.exists(index_path):
                logger.warning(f"Index path '{index_path}' not found, skipping...")
                self.index_path = "" # Сбрасываем путь, если он не найден
            else:
                logger.info(f"Index path: {index_path}")
                self.index_path = index_path
        else:
             self.index_path = ""

        # Initialize the ONNX backend
        self._initialize_backend()

    def _initialize_backend(self):
        """Initialize the ONNX RVC backend"""
        # Check for vec model and download if needed
        # TODO: Сделать имя vec_path динамическим в зависимости от модели RVC (v1/v2)? Пока хардкод.
        self.vec_path = "vec-768-layer-12.onnx"
        vec_local_path = os.path.join(os.getcwd(), self.vec_path)
        # if not os.path.exists(vec_local_path):
        #     logger.info(f"Downloading vec model '{self.vec_path}' for ONNX backend...")
        #     try:
        #         hf_hub_download(
        #             repo_id="NaruseMioShirakana/MoeSS-SUBModel",
        #             filename=self.vec_path,
        #             local_dir=os.getcwd(),
        #             token=False # Используем False вместо None для явности
        #         )
        #     except Exception as e:
        #          logger.error(f"Failed to download vec model: {e}")
        #          raise RuntimeError(f"Failed to download required model file: {self.vec_path}") from e

        try:
            from tts_with_rvc.lib.infer_pack.onnx_inference import OnnxRVC
            # Создаем экземпляр OnnxRVC. Он выполнит всю загрузку ONNX.
            self.onnx_model = OnnxRVC(
                model_path=self.current_model, # Path to the initial RVC model
                vec_path=vec_local_path,       # Path to ContentVec model
                sr=self.sampling_rate,
                hop_size=self.hop_size,
                device=self.device
            )
            logger.info(f"ONNX backend (OnnxRVC instance) initialized with device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize ONNX backend: {e}")
            self.onnx_model = None
            raise RuntimeError(f"Failed to initialize ONNX backend: {e}") from e

    def _start_background_loop(self):
        """Запускает цикл событий в отдельном потоке"""
        if self._loop_thread is not None and self._loop_thread.is_alive():
            return

        def _run_loop(loop):
            asyncio.set_event_loop(loop)
            try:
                loop.run_forever()
            finally:
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()

        self._loop_thread = threading.Thread(target=_run_loop, args=(self.loop,), daemon=True)
        self._loop_thread.start()

    def set_device(self, device):
        if device != self.device:
            logger.info(f"Changing ONNX device from '{self.device}' to '{device}'")
            old_device = self.device
            self.device = device
            try:
                if self.onnx_model is not None:
                    del self.onnx_model # Удаляем старый инстанс перед реинициализацией
                    self.onnx_model = None
                self._initialize_backend() # Полная реинициализация для смены device
                logger.info(f"Successfully changed device to '{self.device}' and re-initialized backend.")
            except Exception as e:
                 logger.error(f"Failed to re-initialize backend for device '{device}': {e}")
                 self.device = old_device # Попытка отката
                 # Может потребоваться более сложная логика восстановления или просто ошибка
                 raise RuntimeError(f"Failed to set device to '{device}'") from e

    def set_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if self.onnx_model is None:
             raise RuntimeError("Backend (OnnxRVC) is not initialized. Cannot set model.")

        current_loaded_path = self.onnx_model.current_rvc_model_path
        if model_path != current_loaded_path:
            logger.info(f"Changing RVC model from '{current_loaded_path}' to '{model_path}'")
            try:
                # Делегируем перезагрузку модели экземпляру OnnxRVC
                self.onnx_model.load_new_rvc_model(model_path)
                self.current_model = model_path # Обновляем путь для справки
                logger.info(f"Successfully changed RVC model to '{self.current_model}'")
            except Exception as e:
                logger.error(f"Failed to load new RVC model '{model_path}': {e}")
                # self.current_model НЕ обновляем, т.к. загрузка не удалась
                raise RuntimeError(f"Failed to set model to '{model_path}'") from e
        else:
            logger.debug(f"Model '{model_path}' is already loaded.")

    def set_voice(self, voice):
        self.current_voice = voice

    def set_index_path(self, index_path):
        if index_path == "":
             logger.info("Index path cleared.")
             self.index_path = ""
        elif not os.path.exists(index_path):
            logger.warning(f"Index path '{index_path}' not found, skipping...")
            self.index_path = "" # Сбрасываем, если не найден
        else:
            logger.info(f"Index path set to: {index_path}")
            self.index_path = index_path

    def set_output_directory(self, directory_path):
        self.output_directory = directory_path

    def _run_async(self, coro):
        """Выполняет корутину в выделенном цикле событий"""
        if not self.loop.is_running():
             logger.warning("Asyncio loop is not running. Starting it.")
             self._start_background_loop() 

             import time
             time.sleep(0.1)
             if not self.loop.is_running():
                  raise RuntimeError("Failed to start asyncio loop.")

        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            return future.result(timeout=300)
        except concurrent.futures.TimeoutError:
            logger.error("Async operation timed out.")
            raise TimeoutError("Async operation timed out.")
        except Exception as e:
             logger.error(f"Error running async task: {e}")
             raise e

    async def _process_with_backend(self, input_path, pitch=0, output_filename=None,
                                    index_rate=0.75, f0method=None, file_index2="",
                                    filter_radius=3, resample_sr=0, rms_mix_rate=0.5,
                                    protect=0.33, verbose=False):
        """Process the audio with ONNX backend"""
        if self.onnx_model is None:
            raise RuntimeError("ONNX backend is not initialized.")

        # Use instance f0_method if not provided for this call
        if f0method is None:
            f0method = self.f0_method

        # Prepare output path
        resolved_output_path = resolve_output_path(self.output_directory, output_filename)

        # Process using ONNX model - передаем все параметры
        audio_opt = self.onnx_model.inference(
            raw_path=input_path,
            sid=0,
            f0_method=f0method,
            f0_up_key=pitch,
            index_file=self.index_path, 
            index_file2=file_index2,
            index_rate=index_rate,
            filter_radius=filter_radius,
            resample_sr=resample_sr,
            rms_mix_rate=rms_mix_rate,
            protect=protect,
            verbose=verbose
            # cr_threshold и pad_time
        )

        target_sr = self.sampling_rate if resample_sr == 0 else resample_sr

        try:
            soundfile.write(resolved_output_path, audio_opt, target_sr)
            logger.info(f"ONNX processed audio saved to: {resolved_output_path}")
        except Exception as e:
            logger.error(f"Failed to write output audio file: {e}")
            raise RuntimeError(f"Failed to write output audio file: {e}") from e

        return os.path.abspath(resolved_output_path)

    def __call__(self,
                 text,
                 pitch=0,
                 tts_rate=0,
                 tts_volume=0,
                 tts_pitch=0,
                 output_filename=None,
                 index_rate=0.75,
                 # is_half убран
                 f0method=None,  
                 file_index2="",  
                 filter_radius=3,  
                 resample_sr=0,  
                 rms_mix_rate=0.5,  
                 protect=0.33,  
                 verbose=False):  

        path = self._run_async(self._speech(
            text=text,
            pitch=pitch,
            tts_add_rate=tts_rate,
            tts_add_volume=tts_volume,
            tts_add_pitch=tts_pitch,
            output_filename=output_filename,
            index_rate=index_rate,
            f0method=f0method, # Передаем
            file_index2=file_index2, # Передаем
            filter_radius=filter_radius, # Передаем
            resample_sr=resample_sr, # Передаем
            rms_mix_rate=rms_mix_rate, # Передаем
            protect=protect, # Передаем
            verbose=verbose # Передаем
        ))

        return path

    async def async_call(self,
                    text,
                    pitch=0,
                    tts_rate=0,
                    tts_volume=0,
                    tts_pitch=0,
                    output_filename=None,
                    index_rate=0.75,
                    # is_half убран
                    f0method=None,
                    file_index2="",
                    filter_radius=3, 
                    resample_sr=0, 
                    rms_mix_rate=0.5, 
                    protect=0.33, 
                    verbose=False): 
        """Асинхронный вариант метода __call__"""

        path = await self._speech(
            text=text,
            pitch=pitch,
            tts_add_rate=tts_rate,
            tts_add_volume=tts_volume,
            tts_add_pitch=tts_pitch,
            output_filename=output_filename,
            index_rate=index_rate,
            f0method=f0method, 
            file_index2=file_index2, 
            filter_radius=filter_radius, 
            resample_sr=resample_sr,
            rms_mix_rate=rms_mix_rate, 
            protect=protect, 
            verbose=verbose
        )

        return path

    async def _speech(self,
                text,
                pitch=0,
                tts_add_rate=0,
                tts_add_volume=0,
                tts_add_pitch=0,
                output_filename=None,
                index_rate=0.75,
                f0method=None,
                file_index2="",
                filter_radius=3, 
                resample_sr=0, 
                rms_mix_rate=0.5, 
                protect=0.33, 
                verbose=False):

        # Generate audio with edge-tts
        input_path, file_name = await tts_communicate(
            tmp_directory=self.tmp_directory, 
            text=text,
            voice=self.current_voice,
            tts_add_rate=tts_add_rate,
            tts_add_volume=tts_add_volume,
            tts_add_pitch=tts_add_pitch
        )

        # Process with ONNX backend
        output_path = await self._process_with_backend(
            input_path=input_path,
            pitch=pitch,
            output_filename=output_filename,
            index_rate=index_rate,
            f0method=f0method, 
            file_index2=file_index2,  
            filter_radius=filter_radius,  
            resample_sr=resample_sr,  
            rms_mix_rate=rms_mix_rate,  
            protect=protect,  
            verbose=verbose  
        )

        # Delete temporary input file
        try:
            os.remove(input_path)
        except OSError as e:
            logger.warning(f"Could not remove temporary TTS file {input_path}: {e}")

        return output_path

    def voiceover_file(self,
                     input_path,
                     pitch=0,
                     output_filename=None,
                     index_rate=0.75,
                     # is_half убран
                     f0method=None,  
                     file_index2="",  
                     filter_radius=3,  
                     resample_sr=0,  
                     rms_mix_rate=0.5,  
                     protect=0.33,  
                     verbose=False):  
        """
        Обрабатывает существующий аудиофайл через RVC (ONNX) без использования TTS.

        Args:
            input_path (str): Путь к входному аудиофайлу для конвертации
            pitch (int): Изменение высоты голоса (в полутонах)
            output_filename (str, optional): Имя выходного файла
            index_rate (float): Скорость индексирования, по умолчанию 0.75
            f0method (str, optional): Метод F0 ('pm', 'dio', 'harvest', 'crepe', 'rmvpe'). По умолчанию используется self.f0_method.
            file_index2 (str, optional): Путь ко второму файлу индекса.
            filter_radius (int, optional): Радиус медианного фильтра для F0. >=3 уменьшает "дыхание".
            resample_sr (int, optional): Частота дискретизации для ресемплинга. 0 - без ресемплинга.
            rms_mix_rate (float, optional): Смешивание RMS огибающей (0-1). Ближе к 0 - ближе к оригиналу.
            protect (float, optional): Защита согласных/дыхания (0-1). Меньше - сильнее защита. 0.5 - выключено.
            verbose (bool, optional): Включить подробное логирование.

        Returns:
            str: Путь к сконвертированному аудиофайлу
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Входной аудиофайл не найден: {input_path}")

        # Процесс через ONNX бэкенд
        return self._run_async(self._process_with_backend(
            input_path=input_path,
            pitch=pitch,
            output_filename=output_filename,
            index_rate=index_rate,
            f0method=f0method,  
            file_index2=file_index2,  
            filter_radius=filter_radius,  
            resample_sr=resample_sr,  
            rms_mix_rate=rms_mix_rate,  
            protect=protect,  
            verbose=verbose  
        ))

    def process_args(self, text):
        rate_param, text = process_text(text, param="--tts-rate")
        volume_param, text = process_text(text, param="--tts-volume")
        tts_pitch_param, text = process_text(text, param="--tts-pitch")
        rvc_pitch_param, text = process_text(text, param="--rvc-pitch")
        return [rate_param, volume_param, tts_pitch_param, rvc_pitch_param], text

    def __del__(self):
        """Останавливает фоновый цикл событий при уничтожении объекта"""
        if hasattr(self, 'loop') and self.loop and self.loop.is_running():
            try:
                self.loop.call_soon_threadsafe(self.loop.stop)
            except RuntimeError as e:
                 logger.warning(f"Error stopping loop: {e}") # Может возникнуть, если цикл уже остановлен
        if hasattr(self, '_loop_thread') and self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=1.0)
        if hasattr(self, 'onnx_model') and self.onnx_model is not None:
             del self.onnx_model
             self.onnx_model = None


def date_to_short_hash():
    current_date = datetime.now()
    # Используем микросекунды для большей уникальности
    date_str = current_date.strftime("%Y-%m-%d %H:%M:%S.%f")
    sha256_hash = hashlib.sha256(date_str.encode()).hexdigest()
    # 8 символов достаточно для временных файлов
    short_hash = sha256_hash[:8]
    return short_hash

# Функция resolve_output_path вынесена для использования в _process_with_backend
def resolve_output_path(output_dir_path, output_filename):
    """Определяет полный путь для выходного файла."""
    # Если указан абсолютный путь в output_filename, используем его
    if output_filename and os.path.isabs(output_filename):
        # Убедимся, что директория существует
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        return output_filename

    # Используем имя файла из output_filename или генерируем по умолчанию
    filename = output_filename if output_filename else (date_to_short_hash() + ".wav")

    # Используем output_dir_path, если он указан
    if output_dir_path:
        os.makedirs(output_dir_path, exist_ok=True)
        return os.path.join(output_dir_path, filename)

    # По умолчанию сохраняем в папку 'temp' в текущей директории
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    return os.path.join(temp_dir, filename)


async def tts_communicate(tmp_directory, # Изменен параметр
                 text,
                 voice="ru-RU-DmitryNeural",
                 tts_add_rate=0,
                 tts_add_volume=0,
                 tts_add_pitch=0):

    if not os.path.exists(tmp_directory):
        os.makedirs(tmp_directory)

    communicate = tts.Communicate(text=text,
                                  voice=voice,
                                  rate=f'{"+" if tts_add_rate >= 0 else ""}{tts_add_rate}%',
                                  volume=f'{"+" if tts_add_volume >= 0 else ""}{tts_add_volume}%',
                                  pitch=f'{"+" if tts_add_pitch >= 0 else ""}{tts_add_pitch}Hz')
    file_name = date_to_short_hash() + ".wav" # Всегда .wav для совместимости
    input_path = os.path.join(tmp_directory, file_name)

    try:
        await communicate.save(input_path)
        logger.info(f"TTS audio saved temporarily to: {input_path}")
    except Exception as e:
        logger.error(f"Failed to save TTS audio: {e}")
        raise RuntimeError(f"Failed to save TTS audio: {e}") from e

    return input_path, file_name


def process_text(input_text, param, default_value=0):
    try:
        words = input_text.split()
        value = default_value
        i = 0

        while i < len(words):
            if words[i] == param:
                if i + 1 < len(words):
                    next_word = words[i + 1]
                    # Проверяем, является ли следующее слово числом (целым или отрицательным)
                    if next_word.lstrip('-').isdigit():
                        value = int(next_word)
                        # Удаляем параметр и его значение из списка слов
                        words.pop(i)
                        words.pop(i)
                        # Так как мы удалили два элемента, не нужно инкрементировать i
                        continue # Переходим к следующей итерации цикла
                    else:
                        # Если значение не число, логируем предупреждение и игнорируем параметр
                        logger.warning(f"Invalid value '{next_word}' for parameter '{param}'. Expected an integer. Ignoring.")
                        # Удаляем только параметр, оставляя возможное "значение" как часть текста
                        words.pop(i)
                        continue # Переходим к следующей итерации
                else:
                    # Если после параметра нет слова, логируем и удаляем параметр
                    logger.warning(f"No value provided for parameter '{param}'. Ignoring.")
                    words.pop(i)
                    continue # Переходим к следующей итерации
            i += 1 # Инкрементируем i, только если не удаляли элементы

        final_string = ' '.join(words)
        return value, final_string

    except Exception as e:
        logger.error(f"Error processing text arguments: {e}")
        return default_value, input_text # Возвращаем исходный текст в случае ошибки