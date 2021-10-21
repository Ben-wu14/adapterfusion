# Adapterfusion pruning with lottery ticket hypothesis 

1. Using Lottery ticket hypothesis to prune the adapter model (weight, neuron, layer)
1. Test the importance of adapter connection by IMP criterion
1. Prune adapter layer using IMP criterion
1. (optional) Use knowledge distillation in adapterfusion to enhance target adapter

Codes:
1. convert.py : convert ben created adapter file to adapter-transformers supported format
2. test_af.py : Training adapterfusion with adapters load from the directiory (adapters_for_af / adapters_for_af_base). Save IMP in file
3. Load_Adapter.ipynb : Main testing program in the project. Include:
    1. Setting task, load dataset and metric
    2. Create new single adapter model
    3. Create adapterfusion with adapter stored in directory
    4. Training and evaluating model with trainer
    5. Setting IMP function and add forward hooks to model
    6. Runing eval dataset to calculate the mean of each IMP, then print it out
    7. Tensorboard setup for model graph visualization
