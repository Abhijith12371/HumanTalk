from nlp_model.llm_chat import ConversationalLLM
from utils.logger import setup_logger
import argparse

logger = setup_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train the conversational LLM")
    parser.add_argument("--dataset", required=True, help="Path to training dataset")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--output_dir", default="./models/llm/finetuned", help="Output directory")
    args = parser.parse_args()
    
    logger.info("Starting LLM training...")
    
    try:
        # Initialize model
        llm = ConversationalLLM()
        
        # Fine-tune on conversational data
        llm.fine_tune(
            dataset_path=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=args.output_dir
        )
        
        logger.info("LLM training completed successfully")
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()