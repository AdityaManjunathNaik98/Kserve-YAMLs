import torch
import torch.nn as nn
import argparse
import pickle
import json
import os
import numpy as np
import random
from collections import defaultdict
from nesy_factory.GNNs import create_model
from nesy_factory.utils.data_utils import load_graph, load_test_queries_by_formula, Query
from nesy_factory.utils import mpqeutils


def convert_results_to_json(results, top_k):
    """
    Convert infer_step results to the JSON format expected by the original inference script.
    
    Args:
        results: Results from model.infer_step()
        top_k: Number of top predictions requested
        
    Returns:
        List of prediction dictionaries in the expected JSON format
    """
    all_predictions = []
    query_id = 1
    
    for query_type in results:
        for formula in results[query_type]:
            for query_result in results[query_type][formula]:
                try:
                    # Extract query information
                    query = query_result['query']
                    
                    # Build prediction dictionary
                    prediction = {
                        "query_id": query_id,
                        "query_type": str(query.formula.query_type),
                        "anchor_nodes": [int(x) for x in query.anchor_nodes],
                        "true_target": int(query_result['true_target']),
                        "target_mode": str(query.formula.target_mode),
                        "true_target_score": float(query_result['true_score']),
                        "true_target_rank": query_result['true_rank'],
                        "total_candidates": query_result['total_candidates'],
                        "top_predictions": [
                            {
                                "rank": i + 1,
                                "node_id": int(node_id),
                                "score": float(score),
                                "is_true_target": node_id == query_result['true_target']
                            }
                            for i, (node_id, score) in enumerate(query_result['top_predictions'][:top_k])
                        ]
                    }
                    
                    all_predictions.append(prediction)
                    query_id += 1
                    
                except Exception as e:
                    # Handle errors gracefully
                    error_result = {
                        "query_id": query_id,
                        "error": str(e),
                        "query_type": str(query.formula.query_type) if hasattr(query, 'formula') else "unknown"
                    }
                    all_predictions.append(error_result)
                    query_id += 1
    
    return all_predictions


def run_inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained_model', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--config_updated', type=str, required=True)
    parser.add_argument('--num_queries', type=str, default="5", help="Number of queries to run inference on")
    parser.add_argument('--top_k', type=str, default="3", help="Number of top predictions to display")
    parser.add_argument('--predictions', type=str, required=True, help="Output path for predictions JSON")
    parser.add_argument('--show_details', type=bool, default=True, help="Show detailed inference output")
    args = parser.parse_args()

    print(f"Received trained_model: {args.trained_model}")
    print(f"Received data_path: {args.data_path}")
    print(f"Received config_updated: {args.config_updated}")
    print(f"Received num_queries: {args.num_queries}")
    print(f"Received top_k: {args.top_k}")
    print(f"Received predictions: {args.predictions}")

    # Load configuration
    try:
        config_path = os.path.join(args.config_updated, 'config_updated.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: {config_path} is not a valid JSON file or is empty.")
        exit(1)
    except FileNotFoundError:
        print(f"Error: {config_path} not found.")
        exit(1)

    # Load and setup model
    print("Loading model...")
    model_name = config.get('model_name', 'mpqe')
    model = create_model(model_name, config)

    if model_name.lower() in ['mpqe', 'rgcn_encoder_decoder']:
        graph, feature_modules, node_maps = load_graph(config['data_dir'], config['embed_dim'])
        if config.get('use_cuda', False) and torch.cuda.is_available():
            graph.features = mpqeutils.cudify(feature_modules, node_maps)
            for key in node_maps:
                node_maps[key] = node_maps[key].cuda()

        out_dims = {mode: config['embed_dim'] for mode in graph.relations}
        enc = mpqeutils.get_encoder(config.get('depth', 0), graph, out_dims, feature_modules, config.get('use_cuda', False))
        model.set_graph_and_encoder(graph, enc)

    device = torch.device('cuda' if config.get('use_cuda', False) and torch.cuda.is_available() else 'cpu')

    # Load model weights
    try:
        model_files = [
            'model.pt', 
            'base_model.pt', 
            'trained_model.pt',
            'model_naive.pt',
            'model_replay.pt', 
            'model_regularized.pt'
        ]
        model_loaded = False
        
        print("Looking for model files in:", args.trained_model)
        if os.path.exists(args.trained_model):
            available_files = os.listdir(args.trained_model)
            print("Available files:", available_files)
            
            pt_files = [f for f in available_files if f.endswith('.pt')]
            if pt_files:
                model_files.extend(pt_files)
                print("Found .pt files:", pt_files)
        
        for model_file in model_files:
            model_path = os.path.join(args.trained_model, model_file)
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=device))
                model = model.to(device)
                print("Loaded Model from", model_path)
                model_loaded = True
                break
        
        if not model_loaded:
            print("Error: No model file found in", args.trained_model)
            if os.path.exists(args.trained_model):
                print("Available files:", os.listdir(args.trained_model))
            else:
                print("Directory does not exist")
            exit(1)
                
    except Exception as e:
        print(f"Error loading weights: {e}")
        exit(1)

    # Load test queries
    print("Loading test queries...")
    try:
        data_file = os.path.join(config['data_dir'], "all_data.pkl")
        if not os.path.exists(data_file):
            data_file = os.path.join(args.data_path, "all_data.pkl")
        
        with open(data_file, "rb") as f:
            all_data = pickle.load(f)
            
        test_queries_raw = []
        query_keys = ['test_queries_2', 'val_queries_2', 'test_queries_3', 'val_queries_3', 'test_edges']
        
        num_queries = int(args.num_queries)
        for key in query_keys:
            if key in all_data:
                queries_data = all_data[key][:num_queries]
                test_queries_raw.extend(queries_data)
                break
        
        if not test_queries_raw:
            print("No test queries found!")
            exit(1)
        
        test_queries = [Query.deserialize(raw_q, keep_graph=True) for raw_q in test_queries_raw]
        print(f"Loaded {len(test_queries)} test queries")
        
    except Exception as e:
        print(f"Error loading test queries: {e}")
        exit(1)

    # Check if model has infer_step method
    if not hasattr(model, 'infer_step'):
        print(f"Error: Model {model_name} does not have infer_step method")
        print("Available methods:", [method for method in dir(model) if not method.startswith('_')])
        exit(1)

    # Run inference using infer_step method
    print("Starting Inference using infer_step method...")
    try:
        top_k = int(args.top_k)
        show_details = args.show_details
        
        # Use the model's infer_step method
        results = model.infer_step(
            queries=test_queries, 
            top_k=top_k, 
            show_details=show_details
        )
        
        # Convert results to expected JSON format
        all_predictions = convert_results_to_json(results, top_k)
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # Save predictions to JSON file
    try:
        os.makedirs(args.predictions, exist_ok=True)
        predictions_file = os.path.join(args.predictions, 'predictions.json')
        
        final_output = {
            "total_queries": len(test_queries),
            "successful_queries": len([p for p in all_predictions if "error" not in p]),
            "failed_queries": len([p for p in all_predictions if "error" in p]),
            "num_queries_requested": num_queries,
            "top_k": top_k,
            "model_used": model_name,
            "inference_method": "infer_step",
            "predictions": all_predictions
        }
        
        with open(predictions_file, 'w') as f:
            json.dump(final_output, f, indent=2)
        
        print("\nInference completed successfully!")
        print("Predictions saved to:", predictions_file)
        print("Total queries processed:", len(test_queries))
        print("Successful predictions:", final_output['successful_queries'])
        print("Failed predictions:", final_output['failed_queries'])
        
        # Print summary statistics
        if final_output['successful_queries'] > 0:
            successful_preds = [p for p in all_predictions if "error" not in p]
            avg_rank = np.mean([p['true_target_rank'] for p in successful_preds if p['true_target_rank'] is not None])
            print(f"Average true target rank: {avg_rank:.2f}")
            
            # Count how many true targets are in top-k
            top_k_hits = sum(1 for p in successful_preds if p['true_target_rank'] and p['true_target_rank'] <= top_k)
            hit_rate = top_k_hits / len(successful_preds) * 100
            print(f"Hit rate @ {top_k}: {hit_rate:.1f}% ({top_k_hits}/{len(successful_preds)})")
        
    except Exception as e:
        print(f"Error saving predictions: {e}")
        exit(1)


if __name__ == "__main__":
    run_inference()