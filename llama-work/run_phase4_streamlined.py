#!/usr/bin/env python3
"""
streamlined runner for phase 4 - uses config file for easy parameter modification
"""

import json
import sys
from phase4_optimization_testing import *

def load_config(config_file='phase4_config.json'):
    """load configuration from json file"""
    with open(config_file, 'r') as f:
        return json.load(f)

def run_from_config(config_file='phase4_config.json'):
    """run tests based on configuration file"""
    
    config = load_config(config_file)
    
    print("="*80)
    print("PHASE 4: STREAMLINED OPTIMIZATION TESTING")
    print("="*80)
    print(f"\nloaded configuration from: {config_file}")
    print(f"model: {config['model']['name']}")
    print(f"number of runs per test: {config['testing']['num_runs']}")
    print(f"total tests: {len(config['test_configurations'])}")
    
    # setup
    os.makedirs(config['testing']['save_dir'], exist_ok=True)
    device = torch.device('cuda' if (config['model']['use_gpu'] and torch.cuda.is_available()) else 'cpu')
    print(f"device: {device}")
    
    # load model
    print(f"\nloading model...")
    tokenizer = LlamaTokenizer.from_pretrained(config['model']['name'])
    model = LlamaForCausalLM.from_pretrained(
        config['model']['name'],
        dtype=torch.float16,
        device_map="auto" if device.type == 'cuda' else None,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    
    model_state = {}
    for name, param in model.named_parameters():
        if param.device.type != 'meta':
            model_state[name] = param
    
    print(f"✓ model loaded: {len(model_state)} parameters")
    
    # run tests
    print("\n" + "="*80)
    print("RUNNING TEST SUITE")
    print("="*80)
    
    results = []
    
    for idx, test_config in enumerate(config['test_configurations'], 1):
        print(f"\n[{idx}/{len(config['test_configurations'])}] {test_config['name']}")
        print(f"  {test_config['description']}")
        print(f"  chunk_size: {test_config['chunk_size_elements']} elements")
        print(f"  concurrency: {test_config['concurrency_limit']}")
        print(f"  compression: {test_config['compression']}")
        
        result = run_multiple_times(
            model_state,
            test_config['name'],
            chunk_size=test_config['chunk_size_elements'],
            concurrency=test_config['concurrency_limit'],
            compression=test_config['compression'],
            num_runs=config['testing']['num_runs']
        )
        results.append(result)
    
    # run custom tests if enabled
    if config.get('custom_tests', {}).get('enabled', False):
        print("\n" + "="*80)
        print("RUNNING CUSTOM TESTS")
        print("="*80)
        
        for idx, test_config in enumerate(config['custom_tests']['configurations'], 1):
            print(f"\n[CUSTOM {idx}] {test_config['name']}")
            
            result = run_multiple_times(
                model_state,
                test_config['name'],
                chunk_size=test_config['chunk_size_elements'],
                concurrency=test_config['concurrency_limit'],
                compression=test_config.get('compression'),
                num_runs=config['testing']['num_runs']
            )
            results.append(result)
    
    # save results
    results_file = f"{config['testing']['save_dir']}results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    print(f"\n{'test':<25} {'save (s)':<18} {'load (s)':<18} {'size (gb)':<15}")
    print("-" * 80)
    for r in results:
        print(f"{r['test_name']:<25} "
              f"{r['save_mean']:>6.2f} ± {r['save_std']:>4.2f}    "
              f"{r['load_mean']:>6.2f} ± {r['load_std']:>4.2f}    "
              f"{r['size_mean']:>5.2f} ± {r['size_std']:>4.2f}")
    
    # generate plots
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    generate_plots(results)
    
    print("\n" + "="*80)
    print("PHASE 4 COMPLETE")
    print("="*80)
    print(f"\n✓ results saved to: {results_file}")
    print(f"✓ plots saved to: {config['testing']['save_dir']}optimization_analysis.png")
    print(f"\nto modify parameters, edit: {config_file}")

if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'phase4_config.json'
    run_from_config(config_file)
