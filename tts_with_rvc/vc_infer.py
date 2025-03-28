import os,sys,pdb,torch
now_dir = os.getcwd()
sys.path.append(now_dir)
import sys
import torch
import numpy as np
import logging
from huggingface_hub import hf_hub_download
from scipy.io import wavfile
from tts_with_rvc.infer.vc.modules import VC
from tts_with_rvc.infer.vc.config import Config

from fairseq.data.dictionary import Dictionary


config = Config()
vc = VC(config)
last_model_path = ""

torch.serialization.safe_globals([Dictionary])
torch.serialization.add_safe_globals([Dictionary])

def rvc_convert(model_path,
            f0_up_key=0,
            input_path=None, 
            output_dir_path=None,
            _is_half="False",
            f0method="rmvpe",
            file_index="",
            file_index2="",
            index_rate=1,
            filter_radius=3,
            resample_sr=0,
            rms_mix_rate=0.5,
            protect=0.33,
            verbose=False,
            output_filename = "out.wav"
          ):  
    '''
    Function to call for the rvc voice conversion.  All parameters are the same present in that of the webui

    Args: 
        model_path (str) : path to the rvc voice model you're using
        f0_up_key (int) : transpose of the audio file, changes pitch (positive makes voice higher pitch)
        input_path (str) : path to audio file (use wav file)
        output_dir_path (str) : path to output directory, defaults to parent directory in output folder
        _is_half (str) : Determines half-precision
        f0method (str) : picks which f0 method to use: dio, harvest, crepe, rmvpe (requires rmvpe.pt)
        file_index (str) : path to file_index, defaults to None
        file_index2 (str) : path to file_index2, defaults to None.  #honestly don't know what this is for
        index_rate (int) : strength of the index file if provided
        filter_radius (int) : if >=3: apply median filtering to the harvested pitch results. The value represents the filter radius and can reduce breathiness.
        resample_sr (int) : quality at which to resample audio to, defaults to no resample
        rmx_mix_rate (int) : adjust the volume envelope scaling. Closer to 0, the more it mimicks the volume of the original vocals. Can help mask noise and make volume sound more natural when set relatively low. Closer to 1 will be more of a consistently loud volume
        protect (int) : protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy

    Returns:
        output_file_path (str) : file path and name of tshe output wav file

    '''
    global last_model_path, vc
    if not os.path.exists(os.path.join(os.getcwd(), "rmvpe.pt")):
        hf_hub_download(repo_id="lj1995/VoiceConversionWebUI", filename="rmvpe.pt", local_dir=os.getcwd(), token=False)

    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps:0"
    else:
        print("Cuda or MPS not detected, but they are required")

    if not verbose:
        logging.getLogger('fairseq').setLevel(logging.ERROR)
        logging.getLogger('rvc').setLevel(logging.ERROR)

    is_half = _is_half

    if output_dir_path == None:
        if output_filename != None:
            output_dir = os.getcwd()
            output_file_path = os.path.join(output_dir, output_filename)
        else:
            if not os.path.exists('temp'):
                os.mkdir('temp')
            output_dir_path = "temp"
            output_file_name = "out.wav"
            output_dir = os.getcwd()
            output_file_path = os.path.join(output_dir,output_dir_path, output_file_name)
    else:
        if output_filename != None:
            output_dir = os.getcwd()
            output_file_path = os.path.join(output_dir, output_filename)
        else:
            output_file_name = "out.wav"
            output_file_path = os.path.join(output_dir_path, output_file_name)

    # create_directory(output_dir_path)
    # output_dir = get_path(output_dir_path)

    if(is_half.lower() == 'true'):
        is_half = True
    else:
        is_half = False

    
    if last_model_path == "" or last_model_path != model_path:
        vc.get_vc(model_path)
        last_model_path = model_path
        
    tgt_sr, opt_wav =vc.vc_single(0,input_path,f0_up_key,None,f0method,file_index,file_index2,index_rate,filter_radius,resample_sr,rms_mix_rate,protect)

    wavfile.write(output_file_path, tgt_sr, opt_wav)
    print(f"\nFile finished writing to: {output_file_path}")

    return output_file_path


def main():
    # Need to comment out yaml setting for input audio
    rvc_convert(model_path="models\\DenVot.pth", input_path="out.wav")

if __name__ == "__main__":
    main()