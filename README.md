# Template
Template for running TAIDE model.

## Install Dependencies
```
pip install -r requirements.txt
```

## Run the Model
```
python main.py
```

## Run Latency Evaluation
Now supporting ttft (prefilling stage) and tpot (decoding stage)
```
python profile_llama.py \
--prompt_len 512 1024 2048 --ttft 
```