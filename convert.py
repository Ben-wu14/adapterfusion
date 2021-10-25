from datasets import load_from_disk, load_metric, load_dataset
from transformers import AutoTokenizer
from transformers import BertConfig, BertModelWithHeads
from transformers import HoulsbyConfig
import pickle
import numpy as np
from transformers import TrainingArguments, Trainer, EvalPrediction



def key_transfer(key, task):
    key = key.replace('downProject','adapter_down.0')
    key = key.replace('upProject','adapter_up')
    key = key.replace('dense.adapter','adapters.'+task)
    key = key.replace('model.',"")
    key = key.replace('classifier','heads.'+task+'.1')
    return key

def transfer_adapter(filePath, task, drop_layer = True):
    with open(filePath,'rb') as file:
        adapter = pickle.load(file)

    new_adapter = {}
    loc_lst = [[False, False] for _ in range(12)]
    leave_out_tuple = (None, None)

    for name, item in adapter.items():
        new_adapter[key_transfer(name, task)] = item
        if drop_layer and 'layer' in name:
            if 'attention' in name:
                layer_id = name[(name.index('layer.')+ len('layer.')) : name.index('.attention')]
                loc_lst[int(layer_id)][0] = True
            else:
                layer_id = name[(name.index('layer.')+ len('layer.')) : name.index('.output')]
                loc_lst[int(layer_id)][1] = True
    if drop_layer:
        mh_leave_out, out_leave_out = [], []
        for i in range(12):
            if not loc_lst[i][0]:
                mh_leave_out.append(i)
            if not loc_lst[i][1]:
                out_leave_out.append(i)
        leave_out_tuple = (mh_leave_out, out_leave_out)

    af_adapter_path = filePath.split('/')[-1].split('.')[0]
    af_adapter_path = 'adapter_param/base/'+af_adapter_path if not drop_layer else 'adapter_param/pruned/adapter/'+af_adapter_path

    with open(af_adapter_path+ '_AF.pkl', 'wb') as f:
        pickle.dump(new_adapter, f)
    return leave_out_tuple, new_adapter

tasks = ['cola','mnli-mm', 'stsb', 'sst2', 'rte', 'mnli', 'qnli', 'qqp', 'mrpc']

for task in tasks:
    is_super_glue = False
    use_prune_adapter = True

    model_checkpoint = '../Code/bert2'

    if is_super_glue:
        dataset_cache_dir = '/mnt/Code/super_glue/'+task
        metric_script = '/mnt/Code/util/super_glue.py'
    else:
        dataset_cache_dir = '/mnt/Code/dataset/'+task
        metric_script = '/mnt/Code/util/glue.py'

    model_checkpoint = '/mnt/Code/bert2'

    actual_task = "mnli" if task == "mnli-mm" else task

    metric = load_metric(metric_script, actual_task, cache_dir = './dataset')
    obtain_data = load_from_disk(dataset_cache_dir)


    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mnli-mm": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp":  ("question1", "question2"),
        "rte":  ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
        'cb':   ("premise", "hypothesis"),
    }
    sentence1_key, sentence2_key = task_to_keys[task]


    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,use_fast=True)

    def encode_batch(examples):
        """Encodes a batch of input data using the model tokenizer."""
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True)
        return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

    # Encode the input data
    dataset = obtain_data.map(encode_batch, batched=True)

    num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2

    model = BertModelWithHeads.from_pretrained(
        model_checkpoint,
    )

    task_to_path = {
        'qnli':'qnli_iter4_adapter.pkl',
        'mnli':'mnli_iter5_adapter.pkl',
        'qqp':'qqp_iter3_adapter.pkl',
        'mrpc':'mrpc_iter9_adapter.pkl',
        'cola':'cola_iter4_adapter.pkl',
        'rte':'rte_iter3_adapter.pkl',
        'sst2':'sst2_iter4_adapter.pkl',
        'stsb':'stsb_iter2_adapter.pkl',
        'mnli-mm':'mnli-mm_iter1_adapter.pkl'
    }
    task_to_base_path = {
        'qnli':'qnli_iter0_adapter.pkl',
        'mnli':'mnli_iter0_adapter.pkl',
        'qqp':'qqp_iter0_adapter.pkl',
        'mrpc':'mrpc_iter0_adapter.pkl',
        'cola':'cola_iter0_adapter.pkl',
        'rte':'rte_iter0_adapter.pkl',
        'sst2':'sst2_iter0_adapter.pkl',
        'stsb':'stsb_iter0_adapter.pkl',
        'mnli-mm':'mnli-mm_iter0_adapter.pkl'
    }

    custom_adapter_path = task_to_path if use_prune_adapter else task_to_base_path

    leave_out_tuple, new_adapter = transfer_adapter( 'ben_adapter/'+custom_adapter_path[task], task, drop_layer = use_prune_adapter)
    mh_leave_out, out_leave_out = leave_out_tuple

    qnli_config = HoulsbyConfig(non_linearity='gelu_orig', reduction_factor=6, adapter_residual_before_ln=True, mh_leave_out=mh_leave_out, out_leave_out=out_leave_out)
    model.add_adapter(task,config=qnli_config, overwrite_ok=True)
    model.train_adapter(task)

    
    model.add_classification_head(task, num_labels=num_labels, layers=1, overwrite_ok=True, use_pooler=True)
    model.set_active_adapters(task)

    notLoadParam, unkownParam = model.load_state_dict(new_adapter, strict=False)

    print("Unknown Param",unkownParam)
    

    metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
    validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"

    training_args = TrainingArguments(
        learning_rate = 5e-5,
        num_train_epochs=4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        logging_steps=50,
        output_dir="/tmp/training_output",
        overwrite_output_dir=True,
        remove_unused_columns=True,
        evaluation_strategy='steps',
        metric_for_best_model=metric_name
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if task !='stsb':
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = predictions[:,0]
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = dataset['train'],
        eval_dataset = dataset[validation_key],
        tokenizer = tokenizer,
        compute_metrics = compute_metrics,

    )
    print(task, metric_name, trainer.evaluate()['eval_'+metric_name])

    save_path = 'adapters_for_af' if use_prune_adapter else 'adapters_for_af_base'

    model.save_all_adapters(save_path, with_head=True)