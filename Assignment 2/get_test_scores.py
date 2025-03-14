import re
import csv

def extract_runs_from_outfile(filename, output_csv):
    runs = []
    with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()
    
    # Split content into individual runs
    run_blocks = re.split(r'(?=Running: lr=)', content)
    
    for block in run_blocks:
        if not block.startswith('Running: lr='):
            continue
        
        # Extract parameters from "Running" line
        params_match = re.search(
            r'Running: lr=([\d.e-]+),\s+batch_size=(\d+),\s+epochs=(\d+),\s+conv_channels=([\d,]+),\s+dropout_rate=([\d.]+),\s+optimizer=(\w+)',
            block
        )
        if not params_match:
            continue
        
        # Extract test metrics
        metrics = {
            'test_acc': re.search(r'test_acc\s+│\s+([\d.]+)\s+│', block),
            'test_f1': re.search(r'test_f1\s+│\s+([\d.]+)\s+│', block),
            'test_loss': re.search(r'test_loss\s+│\s+([\d.]+)\s+│', block),
            'test_precision': re.search(r'test_precision\s+│\s+([\d.]+)\s+│', block),
            'test_recall': re.search(r'test_recall\s+│\s+([\d.]+)\s+│', block)
        }
        
        # Get actual trained epochs from run summary
        epoch_match = re.search(r'wandb: Run summary:\s*\n.*\bepoch (\d+)', block)
        
        run_data = {
            'learning_rate': params_match.group(1),
            'batch_size': params_match.group(2),
            'epochs_trained': epoch_match.group(1) if epoch_match else params_match.group(3),
            'conv_channels': params_match.group(4),
            'dropout_rate': params_match.group(5),
            'optimizer': params_match.group(6),
            **{k: v.group(1) if v else 'N/A' for k, v in metrics.items()}
        }
        runs.append(run_data)
    
    # Filter out runs where all metrics are 'N/A'
    runs = [run for run in runs if any(run[metric] != 'N/A' for metric in ['test_acc', 'test_f1', 'test_loss', 'test_precision', 'test_recall'])]

    if len(runs) > 0:
        # Write to CSV
        with open(output_csv, 'w', newline='') as csvfile:
            fieldnames = [
                'learning_rate', 'batch_size', 'epochs_trained', 'conv_channels',
                'dropout_rate', 'optimizer', 'test_acc', 'test_f1', 'test_loss',
                'test_precision', 'test_recall'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(runs)
        
        print(f"Extracted {len(runs)} runs and saved to {output_csv}")


for run_number in range(10223748, 10499707):
    try:
        extract_runs_from_outfile(f"slurm_output_{run_number}.out", f"test_scores_{run_number}.csv")
    except:
        pass