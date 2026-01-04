"""Federated Cross-Modal Simulation: Server App"""

import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp

from farm.task import Net, load_test_data, test_final
from farm.custom_strategy import FedAvgWithSimilarity

# Create ServerApp
app = ServerApp()


def plot_similarities(similarity_data, rmse_data, num_rounds, local_epochs, missing_config, filename):
    """
    Plot similarity vs epoch with vertical lines at round boundaries.
    
    Args:
        similarity_data: dict with keys 'image' and 'text', normalized similarity values
        rmse_data: dict with keys 'image' and 'text', raw RMSE values  
        num_rounds: number of federated rounds
        local_epochs: epochs per round
        missing_config: configuration string
        filename: output filename for the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Determine which modalities have data to plot
    image_sims = [s for s in similarity_data['image'] if s > 0]
    text_sims = [s for s in similarity_data['text'] if s > 0]
    image_rmse = [r for r in rmse_data['image'] if r > 0]
    text_rmse = [r for r in rmse_data['text'] if r > 0]
    
    has_image_data = len(image_sims) > 0
    has_text_data = len(text_sims) > 0
    
    total_epochs = max(len(similarity_data['image']), len(similarity_data['text']))
    epoch_numbers = list(range(1, total_epochs + 1))
    
    # ========== Plot 1: Normalized Similarity ==========
    if has_image_data:
        ax1.plot(epoch_numbers[:len(similarity_data['image'])], similarity_data['image'], 
                color='blue', marker='o', markersize=4, linewidth=2, 
                label='Image Similarity', alpha=0.8)
    
    if has_text_data:
        ax1.plot(epoch_numbers[:len(similarity_data['text'])], similarity_data['text'], 
                color='red', marker='s', markersize=4, linewidth=2, 
                label='Text Similarity', alpha=0.8)
    
    # Add vertical dotted lines at round boundaries
    for round_num in range(1, num_rounds):
        epoch_boundary = round_num * local_epochs
        if epoch_boundary <= total_epochs:
            ax1.axvline(x=epoch_boundary, color='gray', linestyle=':', 
                       linewidth=2, alpha=0.7)
    
    # Formatting for similarity plot
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Normalized Similarity', fontsize=12, fontweight='bold')
    ax1.set_title(f'Normalized Similarity Evolution: {missing_config}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.legend(loc='best', fontsize=10)
    ax1.set_ylim([0, 1.05])
    ax1.set_xlim([0.5, total_epochs + 0.5])
    
    # ========== Plot 2: RMSE Values ==========
    if has_image_data:
        ax2.plot(epoch_numbers[:len(rmse_data['image'])], rmse_data['image'], 
                color='blue', marker='o', markersize=4, linewidth=2, 
                label='Image RMSE', alpha=0.8)
    
    if has_text_data:
        ax2.plot(epoch_numbers[:len(rmse_data['text'])], rmse_data['text'], 
                color='red', marker='s', markersize=4, linewidth=2, 
                label='Text RMSE', alpha=0.8)
    
    # Add vertical dotted lines at round boundaries
    for round_num in range(1, num_rounds):
        epoch_boundary = round_num * local_epochs
        if epoch_boundary <= total_epochs:
            ax2.axvline(x=epoch_boundary, color='gray', linestyle=':', 
                       linewidth=2, alpha=0.7, label='Round Boundary' if round_num == 1 else '')
    
    # Formatting for RMSE plot
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax2.set_title(f'RMSE Evolution: {missing_config}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.legend(loc='best', fontsize=10)
    ax2.set_xlim([0.5, total_epochs + 0.5])
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Similarity and RMSE plot saved to: {filename}")


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    
    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    local_epochs: int = context.run_config["local-epochs"]
    lr: float = context.run_config["lr"]
    alpha: float = context.run_config["alpha"]
    beta: float = context.run_config["beta"]
    num_supernodes: int = context.run_config.get("num-supernodes", 5)
    
    # Parse missing configs from comma-separated string
    missing_configs_str: str = context.run_config.get("missing-configs", "100_image_100_text")
    missing_configs: list = [config.strip() for config in missing_configs_str.split(",")]
    
    # Iterate through all configs
    for missing_config in missing_configs:
        print(f"\n{'='*80}")
        print(f"Starting Federated Learning with Cross-Modal Simulation")
        print(f"{'='*80}")
        print(f"  Config: {missing_config}")
        print(f"  Rounds: {num_rounds}")
        print(f"  SuperNodes: {num_supernodes}")
        print(f"  Local epochs: {local_epochs}")
        print(f"  Fraction train: {fraction_train}")
        print(f"  Learning rate: {lr}")
        print(f"  Alpha (simulation loss): {alpha}")
        print(f"  Beta (classification loss): {beta}")
        print(f"{'='*80}\n")
        
        # Create config record
        config = ConfigRecord({
            "lr": lr,
            "alpha": alpha,
            "beta": beta,
            "missing-config": missing_config
        })
        
        # Get device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Load global model
        global_model = Net(device=device)
        arrays = ArrayRecord(global_model.state_dict())
        
        # Initialize custom FedAvg strategy
        strategy = FedAvgWithSimilarity(
            fraction_train=fraction_train
        )
        
        # Run federated learning for ALL rounds
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            train_config=config,
            evaluate_config=config,
            num_rounds=num_rounds,
        )
        
        # After all rounds, compute final statistics
        print(f"\n{'='*80}")
        print(f"All {num_rounds} rounds completed - Computing final statistics...")
        print(f"{'='*80}")
        
        stats = strategy.get_statistics(num_rounds, local_epochs)
        rmse_data = stats['rmse']
        similarity_data = stats['similarity']
        
        # Verify the list sizes
        expected_size = num_rounds * local_epochs
        print(f"\nFinal list sizes:")
        print(f"  Image: {len(similarity_data['image'])} (expected: {expected_size})")
        print(f"  Text: {len(similarity_data['text'])} (expected: {expected_size})")
        
        # Create filename with all parameters
        model_filename = f"final_model-n_{num_supernodes}-r_{num_rounds}-e_{local_epochs}-{missing_config}.pt"
        results_filename = f"results-n_{num_supernodes}-r_{num_rounds}-e_{local_epochs}-{missing_config}.json"
        plot_filename = f"similarity_plot-n_{num_supernodes}-r_{num_rounds}-e_{local_epochs}-{missing_config}.png"
        
        # Save final model to disk
        print(f"\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, model_filename)
        print(f"Model saved to: {model_filename}")
        
        # Final evaluation on test set
        print("\n" + "="*80)
        print("Loading final model for test set evaluation...")
        print("="*80)
        
        final_model = Net(device=device)
        final_model.load_state_dict(state_dict)
        final_model.to(device)
        
        # Load test data with same missing config
        testloader = load_test_data(missing_config=missing_config)
        
        # Evaluate on test set
        test_metrics = test_final(final_model, testloader, device)
        
        # Print RMSE and similarity summary
        print(f"\n{'='*80}")
        print(f"WEIGHTED RMSE & SIMILARITY SUMMARY")
        print(f"{'='*80}")
        
        if rmse_data['image']:
            non_zero_rmse = [r for r in rmse_data['image'] if r > 0]
            non_zero_sim = [s for s in similarity_data['image'] if s > 0]
            if non_zero_rmse:
                print(f"\nImage Modality (Weighted Average Across All Clients):")
                print(f"  RMSE - Global Min: {stats['global_stats']['image']['min']:.4f}, "
                      f"Max: {stats['global_stats']['image']['max']:.4f}")
                print(f"  RMSE - Initial: {stats['global_stats']['image']['initial']:.4f}, "
                      f"Final: {stats['global_stats']['image']['final']:.4f}")
                print(f"  RMSE - Average: {stats['global_stats']['image']['mean']:.4f}")
                if non_zero_sim:
                    print(f"  Similarity - Initial: {non_zero_sim[0]:.4f}, Final: {non_zero_sim[-1]:.4f}")
                    print(f"  Similarity - Average: {np.mean(non_zero_sim):.4f}")
        
        if rmse_data['text']:
            non_zero_rmse = [r for r in rmse_data['text'] if r > 0]
            non_zero_sim = [s for s in similarity_data['text'] if s > 0]
            if non_zero_rmse:
                print(f"\nText Modality (Weighted Average Across All Clients):")
                print(f"  RMSE - Global Min: {stats['global_stats']['text']['min']:.4f}, "
                      f"Max: {stats['global_stats']['text']['max']:.4f}")
                print(f"  RMSE - Initial: {stats['global_stats']['text']['initial']:.4f}, "
                      f"Final: {stats['global_stats']['text']['final']:.4f}")
                print(f"  RMSE - Average: {stats['global_stats']['text']['mean']:.4f}")
                if non_zero_sim:
                    print(f"  Similarity - Initial: {non_zero_sim[0]:.4f}, Final: {non_zero_sim[-1]:.4f}")
                    print(f"  Similarity - Average: {np.mean(non_zero_sim):.4f}")
        
        print(f"{'='*80}\n")
        
        # Save test results with RMSE and similarity data
        results_summary = {
            'method': 'cross-modal-simulation-federated',
            'config': missing_config,
            'num_supernodes': num_supernodes,
            'num_rounds': num_rounds,
            'local_epochs': local_epochs,
            'alpha': alpha,
            'beta': beta,
            'test_metrics': test_metrics,
            'tracking': {
                'rmse': {
                    'image': rmse_data['image'],
                    'text': rmse_data['text']
                },
                'normalized_similarity': {
                    'image': similarity_data['image'],
                    'text': similarity_data['text']
                },
                'global_stats': stats['global_stats']
            }
        }
        
        with open(results_filename, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"Test results saved to: {results_filename}")
        
        # Generate similarity plot
        if rmse_data['image'] or rmse_data['text']:
            print(f"\nGenerating similarity and RMSE plots...")
            plot_similarities(similarity_data, rmse_data, num_rounds, local_epochs, 
                            missing_config, plot_filename)
        else:
            print(f"\nWarning: No RMSE data collected for plotting")
        
        print(f"\n{'='*80}")
        print(f"Federated Learning Complete for config: {missing_config}")
        print(f"{'='*80}\n")
    
    print(f"\n{'#'*80}")
    print(f"# ALL CONFIGURATIONS COMPLETED!")
    print(f"{'#'*80}\n")