from transformers import AutoProcessor, BarkModel
import scipy
import torch
import time


def main(text: str):
    start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16).to(device)
    model.save_pretrained("/Users/sid/Desktop/SD/TTS/bark-small") 
    processor = AutoProcessor.from_pretrained("suno/bark")

    voice_preset = "v2/en_speaker_6"

    inputs = processor("Hello, my dog is cute", voice_preset=voice_preset)

    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()
    sample_rate = model.generation_config.sample_rate
    scipy.io.wavfile.write("bark_out.wav", rate=sample_rate, data=audio_array)
    end = time.time()
    print(f"Time taken: {end - start}")


if __name__ == "__main__":
    main("Hello, my dog is cute")