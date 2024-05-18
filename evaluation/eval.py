import subprocess

def ret_fid_alpha():
    python_command = "python -m pytorch_fid /mnt/nvme0n1p5/projects/hackathon/CP2/data/mnist/train/Validation/A /mnt/nvme0n1p5/projects/hackathon/CP2/tools/mnist/cond_class_samples/1 --device cuda:0"

    result = subprocess.run(python_command, shell=True, capture_output=True, text=True)

    output_lines = result.stdout.strip().split("\n")
    fid_line = output_lines[0]
    fid_value = (float(fid_line.split(":")[-1].strip()))

    return round(fid_value,2)

def ret_fid_animal():
    python_command = "python -m pytorch_fid /mnt/nvme0n1p5/projects/hackathon/CP2/data/Animals10/raw-img/elefante /mnt/nvme0n1p5/projects/hackathon/CP2/tools/mnist/cond_class_samples/2 --device cuda:0"

    result = subprocess.run(python_command, shell=True, capture_output=True, text=True)

    output_lines = result.stdout.strip().split("\n")
    fid_line = output_lines[0]
    fid_value = (float(fid_line.split(":")[-1].strip()))

    return round(fid_value,2)


if __name__ == '__main__':
    python_command = "python -m pytorch_fid /mnt/nvme0n1p5/projects/hackathon/CP2/data/Animals10/raw-img/elefante /mnt/nvme0n1p5/projects/hackathon/CP2/tools/mnist/cond_class_samples/2 --device cuda:0"

    result = subprocess.run(python_command, shell=True, capture_output=True, text=True)

    output_lines = result.stdout.strip().split("\n")
    print(output_lines)
    fid_line = output_lines[0]
    print(fid_line)
    fid_value = (round(float(fid_line.split(":")[-1].strip())),2)

    # Print the float value
    print(f'FID Score: {fid_value}')


