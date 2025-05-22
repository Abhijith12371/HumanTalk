import os
import json
from tqdm import tqdm
from utils.audio_processing import preprocess_audio
from utils.logger import setup_logger

logger = setup_logger(__name__)

def prepare_dataset(input_dir, output_dir, metadata_file="metadata.csv"):
    """Prepare TTS dataset from raw audio and transcripts"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Process metadata file
    metadata_path = os.path.join(input_dir, metadata_file)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    processed_metadata = []
    
    with open(metadata_path, 'r') as f:
        for line in tqdm(f.readlines(), desc="Processing audio files"):
            try:
                audio_file, transcript = line.strip().split("|")
                audio_path = os.path.join(input_dir, audio_file)
                
                # Preprocess audio
                audio, sr = preprocess_audio(audio_path)
                
                # Save processed audio
                output_path = os.path.join(output_dir, audio_file)
                sf.write(output_path, audio, sr)
                
                # Add to metadata
                processed_metadata.append(f"{audio_file}|{transcript}")
            
            except Exception as e:
                logger.warning(f"Error processing {audio_file}: {str(e)}")
                continue
    
    # Save processed metadata
    with open(os.path.join(output_dir, "processed_metadata.csv"), 'w') as f:
        f.write("\n".join(processed_metadata))
    
    logger.info(f"Dataset preparation complete. Processed {len(processed_metadata)} files.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Input directory with raw data")
    parser.add_argument("--output_dir", required=True, help="Output directory for processed data")
    args = parser.parse_args()
    
    prepare_dataset(args.input_dir, args.output_dir)