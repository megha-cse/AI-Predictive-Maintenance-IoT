"""Run the complete training and evaluation pipeline."""
from src.train_model import train
from src.evaluate_model import evaluate

if __name__ == "__main__":
    print("=" * 58)
    print("AI-POWERED PREDICTIVE MAINTENANCE SYSTEM")
    print("=" * 58)
    train()
    evaluate()
    print("\nPipeline completed. Start the dashboard with: streamlit run app.py")
