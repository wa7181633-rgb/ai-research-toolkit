# Synthetic Data Pipeline & Experiment Tracking

import json

def generate_synthetic_math_data():
    """Generates synthetic math reasoning problems and step-by-step solutions."""
    synthetic_dataset = [
        {
            "instruction": "Solve for x: 3x + 5 = 20",
            "thought": "Subtract 5 from both sides to get 3x = 15, then divide by 3 to get x = 5.",
            "answer": "5"
        },
        {
            "instruction": "If a shirt costs $40 and is on sale for 25% off, what is the final price?",
            "thought": "25% of 40 is 10. 40 - 10 = 30.",
            "answer": "30"
        }
    ]
    
    with open("synthetic_math_data.json", "w") as f:
        json.dump(synthetic_dataset, f, indent=4)
    print("Synthetic math dataset generated successfully.")

if __name__ == "__main__":
    generate_synthetic_math_data()
