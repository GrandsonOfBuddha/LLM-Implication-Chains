#!/bin/bash

# MVP Implication Chains - Setup and Run Script

echo "Setting up MVP Implication Chains project..."

# Create virtual environment
python3 -m venv mvp_env
source mvp_env/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY environment variable not set."
    echo "Please set it with: export OPENAI_API_KEY='your-api-key'"
    echo "You can also add it to your ~/.bashrc or ~/.zshrc file"
fi

echo "Setup complete!"
echo ""
echo "To run the MVP pipeline:"
echo "1. Set your OpenAI API key: export OPENAI_API_KEY='your-key'"
echo "2. Generate dataset: python build_dataset.py"
echo "3. Train classifier: python train_classifier.py"
echo ""
echo "To test individual components:"
echo "- Test chain generation: python -c 'from implication_chains import *; print(create_seed_statements()[:5])'"
echo "- Check installed packages: pip list"