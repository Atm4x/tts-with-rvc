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

# Применяем nest_asyncio для избежания конфликтов с циклами событий
nest_asyncio.apply()

class TTS_RVC:
    def __init__(self, input_directory, model_path, voice="ru-RU-DmitryNeural", index_path="", 
                 f0_method="dio", output_directory=None, sampling_rate=40000, 
                 hop_size=512, device="dml"):
        self.current_voice = voice
        self.input_directory = input_directory
        self.can_speak = True
        self.current_model = model_path
        self.output_directory = output_directory
        self.f0_method = f0_method
        self.sampling_rate = sampling_rate
        self.hop_size = hop_size
        self.device = device  # Default is 'dml' for DirectML
            
        # Создаем выделенный цикл событий для асинхронных операций
        self.loop = asyncio.new_event_loop()
        self._loop_thread = None
        self._start_background_loop()
        
        if(index_path != ""):
            if not os.path.exists(index_path):
                print("Index path not found, skipping...")
            else:
                print("Index path:", index_path)
        self.index_path = index_path

        # Initialize the ONNX backend
        self._initialize_backend()
        
    def _initialize_backend(self):
        """Initialize the ONNX RVC backend"""
        # Check for vec model and download if needed
        self.vec_path = "vec-768-layer-12.onnx"
        if not os.path.exists(os.path.join(os.getcwd(), self.vec_path)):
            print("Downloading vec model for ONNX backend...")
            hf_hub_download(
                repo_id="NaruseMioShirakana/MoeSS-SUBModel", 
                filename=self.vec_path, 
                local_dir=os.getcwd(), 
                token=False
            )
            
        # Import OnnxRVC when using ONNX backend
        try:
            from tts_with_rvc.lib.infer_pack.onnx_inference import OnnxRVC
            self.onnx_model = OnnxRVC(
                self.current_model,
                vec_path=self.vec_path,
                sr=self.sampling_rate,
                hop_size=self.hop_size,
                device=self.device
            )
            print(f"ONNX backend initialized with device: {self.device}")
        except ImportError:
            print("Failed to import OnnxRVC. Make sure the package is installed with [onnx] extras.")
            raise
            
    def _start_background_loop(self):
        """Запускает цикл событий в отдельном потоке"""
        if self._loop_thread is not None:
            return
            
        def _run_loop(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()
            
        self._loop_thread = threading.Thread(target=_run_loop, args=(self.loop,), daemon=True)
        self._loop_thread.start()
    
    def set_device(self, device):
        """Change ONNX device (dml, cpu, etc)"""
        if device != self.device:
            self.device = device
            # Re-initialize the model with the new device
            self._initialize_backend()
    
    def set_voice(self, voice):
        self.current_voice = voice
    
    def set_index_path(self, index_path):
        if not os.path.exists(index_path) and index_path != "":
            print("Index path not found, skipping...")
        else:
            print("Index path:", index_path)
        self.index_path = index_path

    def set_output_directory(self, directory_path):
        self.output_directory = directory_path
    
    def _run_async(self, coro):
        """Выполняет корутину в выделенном цикле событий"""
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()
    
    async def _process_with_backend(self, input_path, pitch=0, output_filename=None, index_rate=0.75):
        """Process the audio with ONNX backend"""
        # Use ONNX backend for processing
        sid = 0  # Speaker ID, not used in current implementation
        
        # Prepare output path
        if output_filename is None:
            name = date_to_short_hash()
            if self.output_directory is None:
                output_directory = "temp"
                if not os.path.exists(output_directory):
                    os.makedirs(output_directory)
            else:
                output_directory = self.output_directory
            output_filename = os.path.join(output_directory, name + ".wav")
        else:
            if self.output_directory is not None:
                output_filename = os.path.join(self.output_directory, output_filename)
        
        # Process using ONNX model
        audio = self.onnx_model.inference(
            input_path, 
            sid, 
            f0_method=self.f0_method, 
            f0_up_key=pitch
        )
        
        soundfile.write(output_filename, audio, self.sampling_rate)
        return os.path.abspath(output_filename)
    
    def __call__(self,
                 text,
                 pitch=0,
                 tts_rate=0,
                 tts_volume=0,
                 tts_pitch=0,
                 output_filename=None,
                 index_rate=0.75):
        
        path = self._run_async(self._speech(
            text=text,
            pitch=pitch,
            tts_add_rate=tts_rate,
            tts_add_volume=tts_volume,
            tts_add_pitch=tts_pitch,
            output_filename=output_filename,
            index_rate=index_rate
        ))
        
        return path
    
    async def async_call(self,
                    text,
                    pitch=0,
                    tts_rate=0,
                    tts_volume=0,
                    tts_pitch=0,
                    output_filename=None,
                    index_rate=0.75):
        """Асинхронный вариант метода __call__"""
        
        path = await self._speech(
            text=text,
            pitch=pitch,
            tts_add_rate=tts_rate,
            tts_add_volume=tts_volume,
            tts_add_pitch=tts_pitch,
            output_filename=output_filename,
            index_rate=index_rate
        )
        
        return path

    async def _speech(self,
                text,
                pitch=0,
                tts_add_rate=0,
                tts_add_volume=0,
                tts_add_pitch=0,
                output_filename=None,
                index_rate=0.75):
        
        # Generate audio with edge-tts
        input_path, file_name = await tts_communicate(
            input_directory=self.input_directory,
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
            index_rate=index_rate
        )
        
        # Delete temporary input file
        os.remove(input_path)
        
        return output_path

    def voiceover_file(self,
                     input_path,
                     pitch=0,
                     output_filename=None,
                     index_rate=0.75):
        """
        Обрабатывает существующий аудиофайл через RVC без использования TTS.
        
        Args:
            input_path (str): Путь к входному аудиофайлу для конвертации
            pitch (int): Изменение высоты голоса (в полутонах)
            output_filename (str, optional): Имя выходного файла
            index_rate (float): Скорость индексирования, по умолчанию 0.75
            
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
            index_rate=index_rate
        ))

    def process_args(self, text):
        rate_param, text = process_text(text, param="--tts-rate")
        volume_param, text = process_text(text, param="--tts-volume")
        tts_pitch_param, text = process_text(text, param="--tts-pitch")
        rvc_pitch_param, text = process_text(text, param="--rvc-pitch")
        return [rate_param, volume_param, tts_pitch_param, rvc_pitch_param], text
        
    def __del__(self):
        """Останавливает фоновый цикл событий при уничтожении объекта"""
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=1.0)


def date_to_short_hash():
    current_date = datetime.now()
    date_str = current_date.strftime("%Y-%m-%d %H:%M:%S")
    sha256_hash = hashlib.sha256(date_str.encode()).hexdigest()
    short_hash = sha256_hash[:8]
    return short_hash


async def tts_communicate(input_directory,
                 text,
                 voice="ru-RU-DmitryNeural",
                 tts_add_rate=0,
                 tts_add_volume=0,
                 tts_add_pitch=0):
    
    if not os.path.exists(input_directory):
        os.makedirs(input_directory)
        
    communicate = tts.Communicate(text=text,
                                  voice=voice,
                                  rate=f'{"+" if tts_add_rate >= 0 else ""}{tts_add_rate}%',
                                  volume=f'{"+" if tts_add_volume >= 0 else ""}{tts_add_volume}%',
                                  pitch=f'{"+" if tts_add_pitch >= 0 else ""}{tts_add_pitch}Hz')
    file_name = date_to_short_hash() + ".wav"
    input_path = os.path.join(input_directory, file_name)
    await communicate.save(input_path)
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
                    if next_word.isdigit() or (next_word[0] == '-' and next_word[1:].isdigit()):
                        value = int(next_word)
                        words.pop(i)
                        words.pop(i)
                    else:
                        raise ValueError(f"Invalid type of argument in \"{param}\"")
                else:
                    raise ValueError(f"There is no value for parameter \"{param}\"")
            i += 1

        final_string = ' '.join(words)
        return value, final_string

    except Exception as e:
        print(f"Ошибка: {e}")
        return 0, input_text