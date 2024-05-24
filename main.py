from flask import Flask, request, jsonify, send_file
from transformers import AutoProcessor, BarkModel
import scipy.io.wavfile
import torch
import numpy as np
import time
import tempfile
import os
import deepspeed

print("CUDA Available: ", torch.cuda.is_available())
print("CUDA Device Count: ", torch.cuda.device_count())
print("CUDA Current Device: ", torch.cuda.current_device())
print("CUDA Device Name: ", torch.cuda.get_device_name(torch.cuda.current_device()))

print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())

os.environ['SUNO_USE_SMALL_MODELS'] = 'True'

app = Flask(__name__)

# Preload the model and processor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16).to(device)
model.eval()
processor = AutoProcessor.from_pretrained("suno/bark")

# Initialize DeepSpeed
ds_engine, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), 
                                                  config_params={"train_batch_size": 1, "fp16": {"enabled": True}})

@app.route('/infer', methods=['POST'])
def infer():
    data = request.json
    text = data.get('text', '')
    voice_preset = "v2/en_speaker_6"
    
    if not text:
        return jsonify({'error': 'Text is required'}), 400

    start = time.time()
    
    # Prepare inputs
    inputs = processor(text, voice_preset=voice_preset)
    inputs = {key: torch.tensor(value).to(device) if isinstance(value, list) else value.to(device) for key, value in inputs.items()}
    
    # Generate audio array
    with torch.no_grad():
        audio_array = ds_engine.module.generate(**inputs)
    
    # Convert audio array to CPU and numpy format
    audio_array = audio_array.cpu().numpy().squeeze()
    
    # Ensure the audio array is in the correct format (float32)
    audio_array = audio_array.astype(np.float32)
    
    # Define the sample rate
    sample_rate = model.generation_config.sample_rate
    end_before_save = time.time()
    time_taken1 = end_before_save - start
    print(time_taken1)
    
    # Use a temporary file to save the audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        temp_wav_path = tmp_file.name
        scipy.io.wavfile.write(temp_wav_path, rate=sample_rate, data=audio_array)
    
    end = time.time()
    time_taken = end - start
    print(time_taken)
    
    # Send the audio file as response
    response = send_file(temp_wav_path, as_attachment=True, download_name="bark_out.wav")
    response.headers['X-Time-Taken'] = str(time_taken)
    
    # Clean up the temporary file
    @response.call_on_close
    def remove_file():
        try:
            os.remove(temp_wav_path)
        except Exception as e:
            app.logger.error(f"Error removing file: {e}")
    
    return response

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
