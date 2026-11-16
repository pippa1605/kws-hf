import os
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline

def run_batch_inference(dataset_dir):
    # Check if GPU is available to speed up execution
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU (cuda:0)' if device == 0 else 'CPU'}")
    
    # 1. Initialize the audio classification pipeline
    print("Loading MIT/ast-finetuned-speech-commands-v2 model...")
    classifier = pipeline(
        "audio-classification", 
        model="MIT/ast-finetuned-speech-commands-v2",
        device=device
    )
    
    # Track statistics
    total_files = 0
    correct_predictions = 0
    
    # Convert path string to Path object
    base_path = Path(dataset_dir)
    
    # 2. Find all directories (each directory name is the ground truth label)
    # Excluding hidden directories or background noise folders if they exist
    label_dirs = [d for d in base_path.iterdir() if d.is_dir() and not d.name.startswith(('_', '.'))]
    
    if not label_dirs:
        print(f"No label directories found in {dataset_dir}. Please ensure the archive was extracted correctly.")
        return

    print(f"Found {len(label_dirs)} label folders. Starting inference...")

    # 3. Loop through each folder (label)
    for label_dir in label_dirs:
        ground_truth = label_dir.name
        print("dir name: ", ground_truth)
        # Find all .wav files in this subfolder
        wav_files = list(label_dir.glob("*.wav"))
        
        if not wav_files:
            continue
            
        print(f"\nProcessing '{ground_truth}' folder ({len(wav_files)} files)...")
        
        # Loop through each audio file with a progress bar
        for wav_path in tqdm(wav_files, desc=f"Testing {ground_truth}"):
            try:
                # Run the model inference
                # pipeline accepts file paths directly and handles 16kHz resampling internally
                predictions = classifier(str(wav_path))
                
                # The pipeline returns top predictions sorted by score descending.
                # Grab the top-1 predicted label name.
                top_prediction = predictions[0]['label']
                print(" top_prediction: ", top_prediction)
                
                # Update statistics
                total_files += 1
                if top_prediction.lower() == ground_truth.lower():
                    correct_predictions += 1
                    
            except Exception as e:
                print(f"Error processing {wav_path.name}: {e}")
                continue

    # 4. Print Summary Report
    if total_files > 0:
        accuracy = (correct_predictions / total_files) * 100
        print("\n" + "="*40)
        print("INFERENCE TESTING COMPLETE")
        print("="*40)
        print(f"Total files evaluated:  {total_files}")
        print(f"Correctly predicted:   {correct_predictions}")
        print(f"Overall Accuracy:      {accuracy:.2f}%")
        print("="*40)
    else:
        print("\nNo .wav files were evaluated.")

if __name__ == "__main__":
    # Change this path to match where your test archive was extracted
    # (e.g., './speech_commands_v2')
    TARGET_DIR = "./speech_commands_v2-small" 
    
    run_batch_inference(TARGET_DIR)
