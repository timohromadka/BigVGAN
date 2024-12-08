# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import MAX_WAV_VALUE
import bigvgan
from bigvgan import BigVGAN as Generator
from transformers import AutoModel, AutoTokenizer

h = None
device = None
torch.backends.cudnn.benchmark = False


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print(f"Loading '{filepath}'")
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def load_huggingface_model(model_name, device):
    """
    Load a model from Hugging Face.
    """
    print(f"Loading Hugging Face model: {model_name}")
    model = bigvgan.BigVGAN.from_pretrained(model_name, use_cuda_kernel=True)
    model.to(device)
    print(f"Model {model_name} loaded successfully.")
    return model


def inference(a, h, generator):
    filelist = os.listdir(a.input_mels_dir)

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        for i, filename in enumerate(filelist):
            # Load the mel spectrogram in .npy format
            x = np.load(os.path.join(a.input_mels_dir, filename))
            x = torch.FloatTensor(x).to(device)
            if len(x.shape) == 2:
                x = x.unsqueeze(0)

            y_g_hat = generator(x)

            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype("int16")

            output_file = os.path.join(
                a.output_dir, os.path.splitext(filename)[0] + "_generated_e2e.wav"
            )
            write(output_file, h.sampling_rate, audio)
            print(output_file)


def main():
    print("Initializing Inference Process..")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_mels_dir", default="test_mel_files")
    parser.add_argument("--output_dir", default="generated_files_from_mel")
    parser.add_argument("--checkpoint_file", required=False, help="Path to local checkpoint file.")
    parser.add_argument("--model_name", required=False, help="Hugging Face model name.")
    parser.add_argument("--use_cuda_kernel", action="store_true", default=False)

    a = parser.parse_args()

    global h, device

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if a.model_name:
        # Load Hugging Face model
        generator = load_huggingface_model(a.model_name, device)
        h = AttrDict({"sampling_rate": 44100})  # Replace with appropriate HF model config if needed
    elif a.checkpoint_file:
        # Load local checkpoint
        config_file = os.path.join(os.path.split(a.checkpoint_file)[0], "config.json")
        with open(config_file) as f:
            data = f.read()
        json_config = json.loads(data)
        h = AttrDict(json_config)

        torch.manual_seed(h.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(h.seed)

        generator = Generator(h, use_cuda_kernel=a.use_cuda_kernel).to(device)
        state_dict_g = load_checkpoint(a.checkpoint_file, device)
        generator.load_state_dict(state_dict_g["generator"])
    else:
        raise ValueError("You must provide either --model_name or --checkpoint_file.")

    inference(a, h, generator)


if __name__ == "__main__":
    main()
