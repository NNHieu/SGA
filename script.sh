# Start server
CUDA_VISIBLE_DEVICES=3 vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype auto --api-key token-abc123 --port 10003 --max-model-len 16384

# llama-3.1-8b
python experiment/script/sr/sr.py --llm_port 10003 --llm_config_path experiment/llm_configs/local_llama3.yaml --ds_name invfeynman --end_idx 190
python experiment/script/sr/sr.py --llm-port 10003 --llm_config_path experiment/llm_configs/local_llama3.yaml --ds_name invfeynman --start_idx 190

# gpt4o_mini
python experiment/script/sr/sr.py --llm_config_path experiment/llm_configs/gpt4o_mini.yaml --ds_name feynman
python experiment/script/sr/sr.py --llm_config_path experiment/llm_configs/gpt4o_mini.yaml --ds_name invfeynman
python experiment/script/sr/sr.py --llm_config_path experiment/llm_configs/gpt4o_mini.yaml --ds_name bio
python experiment/script/sr/sr.py --llm_config_path experiment/llm_configs/gpt4o_mini.yaml --ds_name chem
python experiment/script/sr/sr.py --llm_config_path experiment/llm_configs/gpt4o_mini.yaml --ds_name mat
python experiment/script/sr/sr.py --llm_config_path experiment/llm_configs/gpt4o_mini.yaml --ds_name phy