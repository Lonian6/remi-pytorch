# remi-pytorch
Re-write the pop-music transformer and integrate with hugging face API

# Data preparation
Please follow the step in REMI. https://github.com/YatingMusic/remi

# Training
You can change `--dict_path` to specify whether using chord or not.

## from scratch
`python main.py`

## continue
`python main.py --is_continue True --continue_pth your-path-here`

# Testing
`--dict_path` should be smae as training option.

## generate from scratch
`python main.py --is_train False --prompt False --output_path your-path-here --model_path your-path-here`
                
## continue generating
`python main.py --is_train False --prompt_path your-path-here --output_path your-path-here --model_path your-path-here`
