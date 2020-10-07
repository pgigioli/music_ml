import torch

def print_and_log(string, filename):
    print(string)
    with open(filename, 'a') as write_file:
        write_file.write(string + '\n')
        
def partition_audio(audio, window_sizes, step_size, sample_rate=16000):
    window_sizes = [x*sample_rate for x in window_sizes]
    step_size = step_size*sample_rate
    max_window_size = max(window_sizes)

    frames = []
    for i in range(int(audio.shape[-1]/step_size - int(max_window_size/step_size) + 1)):
        for size in window_sizes:
            frame = audio[..., i*step_size:i*step_size+int(size)]
            frames.append(frame)
    return frames

def mask_tensor(tensor, lens, max_len):
    mask = torch.arange(max_len).expand(len(lens), max_len) < lens.unsqueeze(1)
    mask = mask.unsqueeze(1).unsqueeze(1).expand(tensor.size()).to(torch.float32)
    return tensor*mask