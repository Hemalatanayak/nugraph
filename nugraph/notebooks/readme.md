## Domain Adaptation for Semantic Decoder Setup

To run Domain Adaptation (DA) specifically for the Semantic Decoder, you need to properly configure your YAML file and use the provided training script.

### 1. Configuration (`config.yaml`)

Before running the script, ensure your configuration file is set up for Domain Adaptation with two input files (source and target datasets). 

Update the following key parameters in your `config.yaml`:

```yaml
# Enable Domain Adaptation and the Semantic Decoder
run_domain_adaptation: True
run_semantic_decoder: True

# Set input files to 2 for Source and Target datasets
input_files: 2 

# Define your dataset paths
source_filename: "/path/to/source.h5"
target_filename: "/path/to/target.h5"
```

### 2. Running the Script

To run training, execute the following command:

```bash
python train.py -c config.yaml > training_output.log 2>&1
```

If you want to resume the training from a certain checkpoint, give the correct path in the config and run with the following command:

```bash
python train.py -c config.yaml --resume > training_output.log 2>&1
```

In order to just do the inference, give the correct model path in the config and run the following command:

```bash
python train.py -c config.yaml --inference_only > training_output.log 2>&1
```