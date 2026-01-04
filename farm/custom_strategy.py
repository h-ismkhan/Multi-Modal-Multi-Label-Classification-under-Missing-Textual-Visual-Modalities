"""Custom FedAvg strategy that collects RMSE per round and processes at the end"""

from typing import List, Tuple, Dict, Optional
import numpy as np
from flwr.serverapp.strategy import FedAvg
from flwr.app import Message


class FedAvgWithSimilarity(FedAvg):
    """FedAvg strategy that collects RMSE lists from each round"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store RMSE data from each round
        # Structure: self.all_rounds_data[round_idx] = list of client data
        self.all_rounds_data = []
        
    def aggregate_train(
        self, 
        rnd: int, 
        results: List[Message]
    ) -> Tuple[Optional[Message], Dict]:
        """Aggregate training results and collect RMSE from this round"""
        
        # Call parent's aggregation for model weights
        aggregated_result = super().aggregate_train(rnd, results)
        
        # Collect data from all clients for THIS round
        round_data = []
        
        for msg in results:
            if hasattr(msg, 'content') and msg.content is not None:
                if "metrics" in msg.content:
                    metrics_record = msg.content["metrics"]
                    
                    # Extract the metrics dictionary
                    if hasattr(metrics_record, 'metrics'):
                        metrics = metrics_record.metrics
                    elif hasattr(metrics_record, 'to_dict'):
                        metrics = metrics_record.to_dict()
                    else:
                        metrics = metrics_record
                    
                    # Store this client's RMSE for this round
                    client_data = {
                        'image_rmse': metrics.get("image_rmse", []),
                        'text_rmse': metrics.get("text_rmse", []),
                        'client_size': metrics.get("client_size", 1)
                    }
                    round_data.append(client_data)
        
        # Store this round's data
        self.all_rounds_data.append(round_data)
        
        print(f"\nRound {rnd} completed - Collected RMSE from {len(round_data)} clients")
        
        return aggregated_result
    
    def compute_final_statistics(self, num_rounds, local_epochs):
        """
        Process all collected RMSE data:
        1. Find global min/max across ALL rounds and clients
        2. Normalize all values
        3. Compute weighted average per epoch
        4. Create final list L of size (num_rounds × local_epochs)
        """
        if not self.all_rounds_data:
            return {'image': [], 'text': []}, {'image': [], 'text': []}
        
        # Step 1: Collect ALL RMSE values to find global min/max
        all_image_rmse = []
        all_text_rmse = []
        
        for round_data in self.all_rounds_data:
            for client_data in round_data:
                all_image_rmse.extend([r for r in client_data['image_rmse'] if r > 0])
                all_text_rmse.extend([r for r in client_data['text_rmse'] if r > 0])
        
        # Compute global min/max
        global_stats = {
            'image': {
                'min': min(all_image_rmse) if all_image_rmse else 0,
                'max': max(all_image_rmse) if all_image_rmse else 1
            },
            'text': {
                'min': min(all_text_rmse) if all_text_rmse else 0,
                'max': max(all_text_rmse) if all_text_rmse else 1
            }
        }
        
        print(f"\nGlobal Statistics:")
        if all_image_rmse:
            print(f"  Image RMSE - Min: {global_stats['image']['min']:.4f}, Max: {global_stats['image']['max']:.4f}")
        if all_text_rmse:
            print(f"  Text RMSE - Min: {global_stats['text']['min']:.4f}, Max: {global_stats['text']['max']:.4f}")
        
        # Step 2 & 3: Normalize and compute weighted average
        final_rmse = {'image': [], 'text': []}
        final_similarity = {'image': [], 'text': []}
        
        # Process each round
        for round_idx, round_data in enumerate(self.all_rounds_data):
            print(f"\nProcessing Round {round_idx + 1}...")
            
            # Process each epoch in this round
            for epoch_idx in range(local_epochs):
                # IMAGE modality
                weighted_sum_img = 0.0
                weight_sum_img = 0.0
                
                for client_data in round_data:
                    if epoch_idx < len(client_data['image_rmse']) and client_data['image_rmse'][epoch_idx] > 0:
                        raw_rmse = client_data['image_rmse'][epoch_idx]
                        # Normalize this value
                        if global_stats['image']['max'] > global_stats['image']['min']:
                            normalized = (raw_rmse - global_stats['image']['min']) / \
                                       (global_stats['image']['max'] - global_stats['image']['min'])
                            similarity = 1.0 - normalized
                        else:
                            similarity = 1.0
                        
                        # Weighted sum
                        weighted_sum_img += similarity * client_data['client_size']
                        weight_sum_img += client_data['client_size']
                
                # Compute weighted average for this epoch
                if weight_sum_img > 0:
                    avg_similarity = weighted_sum_img / weight_sum_img
                    final_similarity['image'].append(avg_similarity)
                    # Compute corresponding RMSE (denormalize)
                    denormalized_rmse = (1.0 - avg_similarity) * \
                                       (global_stats['image']['max'] - global_stats['image']['min']) + \
                                       global_stats['image']['min']
                    final_rmse['image'].append(denormalized_rmse)
                else:
                    final_similarity['image'].append(0.0)
                    final_rmse['image'].append(0.0)
                
                # TEXT modality
                weighted_sum_txt = 0.0
                weight_sum_txt = 0.0
                
                for client_data in round_data:
                    if epoch_idx < len(client_data['text_rmse']) and client_data['text_rmse'][epoch_idx] > 0:
                        raw_rmse = client_data['text_rmse'][epoch_idx]
                        # Normalize this value
                        if global_stats['text']['max'] > global_stats['text']['min']:
                            normalized = (raw_rmse - global_stats['text']['min']) / \
                                       (global_stats['text']['max'] - global_stats['text']['min'])
                            similarity = 1.0 - normalized
                        else:
                            similarity = 1.0
                        
                        # Weighted sum
                        weighted_sum_txt += similarity * client_data['client_size']
                        weight_sum_txt += client_data['client_size']
                
                # Compute weighted average for this epoch
                if weight_sum_txt > 0:
                    avg_similarity = weighted_sum_txt / weight_sum_txt
                    final_similarity['text'].append(avg_similarity)
                    # Compute corresponding RMSE (denormalize)
                    denormalized_rmse = (1.0 - avg_similarity) * \
                                       (global_stats['text']['max'] - global_stats['text']['min']) + \
                                       global_stats['text']['min']
                    final_rmse['text'].append(denormalized_rmse)
                else:
                    final_similarity['text'].append(0.0)
                    final_rmse['text'].append(0.0)
        
        return final_rmse, final_similarity, global_stats
    
    def get_statistics(self, num_rounds, local_epochs):
        """Compute and return comprehensive statistics"""
        rmse_history, similarity_history, global_stats = self.compute_final_statistics(num_rounds, local_epochs)
        
        # Add additional stats
        for modality in ['image', 'text']:
            non_zero = [r for r in rmse_history[modality] if r > 0]
            if non_zero:
                global_stats[modality].update({
                    'mean': np.mean(non_zero),
                    'initial': non_zero[0],
                    'final': non_zero[-1]
                })
        
        return {
            'rmse': rmse_history,
            'similarity': similarity_history,
            'global_stats': global_stats
        }