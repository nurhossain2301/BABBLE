import json
import os
import argparse
import glob
from scipy.io.wavfile import read, write

def generate_inference_json_2s(wav_filename, output_dir, segment_duration=2.0, skip=2.0, sample_rate=16000):
    """
    Generates a JSON file for inference, segmenting the WAV file into fixed-duration segments.

    Args:
        wav_filename (str): Path to the WAV file.
        output_dir (str): Directory to save the JSON file.
        segment_duration (float): Duration of each segment in seconds (default: 2.0s).
        skip (float): Duration of each segment for skip (default: 2s).
        sample_rate (int): Sample rate of the audio file (default: 16kHz).

    Returns:
        None: Saves the JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    file_basename = os.path.basename(wav_filename)
    output_json = os.path.join(output_dir, f"inference_data_{file_basename}.json")

    inference_data = {}
    rate, data = read(wav_filename)
    total_samples = int((len(data) / rate) // skip) + 1

    for i in range(total_samples):
        start_sample = max(0, int(i * skip * rate))
        stop_sample = min(start_sample + int(segment_duration * sample_rate), len(data))
        if stop_sample - start_sample < sample_rate:  # Less than 1 second
            continue
        segment_key = f"{file_basename}_{start_sample // rate}_{stop_sample // rate}"

        inference_data[segment_key] = {
            "wav": {
                "file": wav_filename,
                "start": start_sample,
                "stop": stop_sample
            }
        }

    # Save to JSON file
    with open(output_json, "w") as f:
        json.dump(inference_data, f, indent=2)

    print(f"Inference data saved to {output_json}")


def generate_inference_json_finer_resolution(wav_filename, output_dir, segment_duration=10.0, skip=0.1, sample_rate=16000):
    """
    Generates a JSON file for inference, segmenting the WAV file into fixed-duration segments for predicting centered 0.1s.

    Args:
        wav_filename (str): Path to the WAV file.
        output_dir (str): Directory to save the JSON file.
        segment_duration (float): Duration of each segment in seconds (default: 2.0s).
        skip (float): Duration of each segment for skip (default: 0.1s).
        sample_rate (int): Sample rate of the audio file (default: 16kHz).

    Returns:
        None: Saves the JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    file_basename = os.path.basename(wav_filename)
    output_json = os.path.join(output_dir, f"inference_data_fine_{file_basename}.json")

    inference_data = {}
    rate, data = read(wav_filename)
    total_samples = int((len(data) / rate) // skip) + 1

    for i in range(total_samples):
        start_sample = max(0, int((i * skip - 1) * rate))
        stop_sample = min(int((i * skip + 1) * rate), len(data))
        if stop_sample - start_sample < sample_rate:  # Less than 1 second
            continue
        segment_key = f"{file_basename}_{round(i * skip, 1)}_{round(i * skip + 0.1, 1)}"

        inference_data[segment_key] = {
            "wav": {
                "file": wav_filename,
                "start": start_sample,
                "stop": stop_sample
            }
        }

    # Save to JSON file
    with open(output_json, "w") as f:
        json.dump(inference_data, f, indent=2)

    print(f"Inference data saved to {output_json}")


def process_wav_folder(folder_path, output_dir):
    """
    Processes all WAV files in a given folder and generates inference JSON files for each.

    Args:
        folder_path (str): Path to the folder containing WAV files.
        output_dir (str): Directory to save the generated JSON files.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    wav_files = glob.glob(os.path.join(folder_path, "*.wav"))

    if not wav_files:
        print("No WAV files found in the specified folder.")
        return

    for wav_file in wav_files:
        wav_path = os.path.join(folder_path, wav_file)

        print(f"Processing: {wav_path}")
        # generate_inference_json_2s(wav_path, output_dir)
        generate_inference_json_finer_resolution(wav_path, output_dir)


def main():
    """
    Main function to parse command-line arguments and run the script.
    """
    parser = argparse.ArgumentParser(description="Process WAV files and generate inference JSON files.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing WAV files.")
    parser.add_argument("output_dir", type=str, help="Path to the folder where JSON files will be saved.")

    args = parser.parse_args()
    
    process_wav_folder(args.folder_path, args.output_dir)


if __name__ == "__main__":
    main()
