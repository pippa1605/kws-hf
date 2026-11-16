import os
import torch
from pathlib import Path
from transformers import pipeline
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

from tqdm import tqdm
from transformers import pipeline

#################################
# download and save model locally
#################################
def download_save_model():
    model_name = "MIT/ast-finetuned-speech-commands-v2"
    local_save_path = "./local_ast_model"

    print("Downloading and saving model assets locally...")
    # Download and save the processor/feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    feature_extractor.save_pretrained(local_save_path)

    # Download and save the core model weights
    model = AutoModelForAudioClassification.from_pretrained(model_name)
    model.save_pretrained(local_save_path)

    print(f"Successfully saved to: {local_save_path}")
# end

def run_batch_inference(dataset_dir):
    # Check if GPU is available to speed up execution
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU (cuda:0)' if device == 0 else 'CPU'}")
    
    # 1. Initialize the audio classification pipeline
    """
    print("Loading MIT/ast-finetuned-speech-commands-v2 model...")
    classifier = pipeline(
        "audio-classification", 
        model="MIT/ast-finetuned-speech-commands-v2",
        device=device
    )  
    """
    LOCAL_MODEL_PATH = "./local_ast_model"    
    print(f"Loading model strictly from local storage: {LOCAL_MODEL_PATH}")    
    # Pass the local folder path to both the 'model' and 'feature_extractor' arguments
    classifier = pipeline(
        "audio-classification", 
        model=LOCAL_MODEL_PATH,
        feature_extractor=LOCAL_MODEL_PATH,
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
"""
if __name__ == "__main__":
    # Change this path to match where your test archive was extracted
    # (e.g., './speech_commands_v2')
    TARGET_DIR = "./speech_commands_v2-small" 
    
    run_batch_inference(TARGET_DIR)
"""

#################################
# load local model and run inference
#################################
def test_local_inference(dataset_dir):
    device = 0 if torch.cuda.is_available() else -1
    
    # Define the local directory containing your saved weights
    # LOCAL_MODEL_PATH = "./local_ast_model"
    LOCAL_MODEL_PATH =  "./ast_speech_commands_finetuned"
    
    print(f"Loading model strictly from local storage: {LOCAL_MODEL_PATH}")
    
    # Pass the local folder path to both the 'model' and 'feature_extractor' arguments
    classifier = pipeline(
        "audio-classification", 
        model=LOCAL_MODEL_PATH,
        feature_extractor=LOCAL_MODEL_PATH,
        device=device
    )
    print("--- model loaded ---")
    # Perform a quick evaluation test
    # sample_file = Path(dataset_dir) / "yes" / "0bac8a71_nohash_0.wav"
    #sample_file = Path(dataset_dir) / "yes" / "0cb74144_nohash_0.wav"
    #sample_file = Path(dataset_dir) / "yes" / "0cb74144_nohash_1.wav"
    sample_file = Path(dataset_dir) / "yes" / "0cb74144_nohash_2.wav"
    #sample_file = Path(dataset_dir) / "yes" / "0cb74144_nohash_3.wav"
    
    if sample_file.exists():
        result = classifier(str(sample_file))
        print(f"\nTest Result for {sample_file.name}:")
        # print(result)
        print("--- top prediction ---")
        top_prediction = result[0]['label']
        print(" top_prediction: ", top_prediction)
        
    else:
        print(f"\nModel loaded successfully. (Sample file {sample_file} not found to auto-verify)")

if __name__ == "__main__":
    # model download
    # download_save_model()
    
    print("test_local_inference")
    test_local_inference("./speech_commands_v2")
    
    #TARGET_DIR = "./speech_commands_v2-small"     
    #print("run_batch_inference")
    #run_batch_inference(TARGET_DIR)
