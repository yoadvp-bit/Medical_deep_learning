import re
import csv

def extract_runs_from_outfile(filename, output_csv):
    runs = []
    with open(filename, 'r') as file:
        content = file.read()
    
    # Split the content into run blocks using "Running: " as a delimiter
    run_blocks = re.split(r'(?=Running: )', content)
    
    for block in run_blocks:
        if not block.startswith('Running: '):
            continue
        
        # Extract parameters from the "Running: " line
        params_match = re.search(
            r'Running: lr=([\d.]+), batch_size=(\d+), epochs=(\d+), channels=([\d,]+), optimizer=(\w+)',
            block
        )
        if not params_match:
            continue  # Skip if the pattern doesn't match
        
        lr = params_match.group(1)
        batch_size = params_match.group(2)
        epochs_param = params_match.group(3)  # The epochs parameter from the Running line
        channels = params_match.group(4)
        optimizer = params_match.group(5)
        
        # Extract test metrics from the test table
        test_acc_match = re.search(r'test_acc\s+│\s+([\d.]+)\s+│', block)
        test_f1_match = re.search(r'test_f1\s+│\s+([\d.]+)\s+│', block)
        test_loss_match = re.search(r'test_loss\s+│\s+([\d.]+)\s+│', block)
        test_precision_match = re.search(r'test_precision\s+│\s+([\d.]+)\s+│', block)
        test_recall_match = re.search(r'test_recall\s+│\s+([\d.]+)\s+│', block)
        
        # Extract the actual epoch trained from the run summary
        epoch_summary_match = re.search(r'wandb: Run summary:\s*\n.*\bepoch (\d+)', block)
        epoch_trained = epoch_summary_match.group(1) if epoch_summary_match else epochs_param
        
        run_data = {
            'learning_rate': lr,
            'batch_size': batch_size,
            'epoch': epoch_trained,
            'channels': channels,
            'optimizer': optimizer,
            'test_acc': test_acc_match.group(1) if test_acc_match else 'N/A',
            'test_f1': test_f1_match.group(1) if test_f1_match else 'N/A',
            'test_loss': test_loss_match.group(1) if test_loss_match else 'N/A',
            'test_precision': test_precision_match.group(1) if test_precision_match else 'N/A',
            'test_recall': test_recall_match.group(1) if test_recall_match else 'N/A'
        }
        runs.append(run_data)
    
    # Write the extracted data to a CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['learning_rate', 'batch_size', 'epoch', 'channels', 'optimizer',
                      'test_acc', 'test_f1', 'test_loss', 'test_precision', 'test_recall']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(runs)
    
    print(f"Extracted {len(runs)} runs and saved to {output_csv}")

# Example usage
run_number = 10499706  # Replace with your actual run number
extract_runs_from_outfile(f"slurm_output_{run_number}.out", f"test_scores_{run_number}.csv")