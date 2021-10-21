from datasets import load_from_disk, load_metric
from transformers import AutoTokenizer, AutoModelWithHeads, TrainingArguments, Trainer, EvalPrediction
from transformers.adapters.composition import Fuse
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch
import pickle



def get_encode_batch(tokenizer, sentence1_key, sentence2_key):
    def encode_batch(examples):
        """Encodes a batch of input data using the model tokenizer."""
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True)
        return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)
    return encode_batch


def load_leave_out(loc_lst):
    mh_leave_out, out_leave_out = [], []
    for i in range(12):
        if not loc_lst[i][0]:
            mh_leave_out.append(i)
        if not loc_lst[i][1]:
            out_leave_out.append(i)
        
    return mh_leave_out, out_leave_out

def load_adapters(directory, af_adapters, model, drop_layer = False):
    for task in af_adapters:
        if not drop_layer:
            model.load_adapter(directory+'/'+task, with_head=False, overwrite_ok=True)
        else:
            with open('adapter_param/pruned/loc_lst/'+task+'_loc_lst.pkl', 'rb') as f:
                    loc_lst = pickle.load(f)
            mh_leave_out, out_leave_out = load_leave_out(loc_lst)
            model.load_adapter(directory+'/'+task, with_head=False, overwrite_ok=True, mh_leave_out=mh_leave_out, out_leave_out=out_leave_out)


def get_compute_metrics(metric, task):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if task !='stsb':
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = predictions[:,0]
        return metric.compute(predictions=predictions, references=labels)
    return compute_metrics

def importance(dic):
    # values * scores
    values = dic['values']
    scores = dic['scores']
    origin_out = dic['origin_out']
    combined = dic['combined']

    adapters_output = torch.unsqueeze(scores, dim=3) * values

    origin_out = torch.unsqueeze(origin_out, dim=2)
    combined = torch.unsqueeze(combined, dim=3)

    # dot products
    origin_dot_product = torch.matmul(origin_out, combined)
    adapter_dot_product = torch.matmul(adapters_output, combined)
    

    adapter_dot_product = torch.squeeze(adapter_dot_product, dim=3)
    origin_dot_product = torch.squeeze(origin_dot_product, dim=3)


    # importance
    dot_products = torch.cat((origin_dot_product, adapter_dot_product), dim=-1)
    projection_percent = dot_products/ torch.sum(combined**2, dim=2)
    
    return projection_percent.mean(dim=(0,1)).cpu()

def get_af_in_out(name, layer, connections, imp_dict):
    def af_hook(self, input, output):
        origin_out = input[-1].data.detach()
        combined = output.data.detach()
        connections[(name, layer)]['origin_out'] = origin_out 
        connections[(name, layer)]['combined'] = combined
        if 'importance' not in imp_dict[(name, layer)]:
            imp_dict[(name, layer)]['importance'] = importance(connections[(name, layer)]).unsqueeze(dim=-1)
        else:
            imp_dict[(name, layer)]['importance'] = torch.cat((imp_dict[(name, layer)]['importance'], importance(connections[(name, layer)]).unsqueeze(dim=-1)), dim=-1)
    return af_hook

def get_output(name, layer, variable, connections):
    def output_hook(self, input, output):
        connections[(name, layer)][variable] = output.data.detach()
    return output_hook

def add_register(module, name, layer, connections, imp_dict):
    module_dict = module.adapter_fusion_layer
    handle = []
    if len(module_dict.keys())!=0:
        adapters = list(module.adapters.keys())
        connections[(name, layer)]['adapter_names'] = adapters
        imp_dict[(name, layer)]['adapter_names'] = adapters
        af_module = module_dict[list(module_dict.keys())[0]]

        handle_scores  = af_module.softmax.register_forward_hook(get_output(name, layer, 'scores', connections))
        handle_values  = af_module.value.register_forward_hook(get_output(name, layer, 'values', connections))
        handle_af  = af_module.register_forward_hook(get_af_in_out(name, layer, connections, imp_dict))

        handle = [handle_scores, handle_values, handle_af]

    return handle


def test(task, af_adapters, is_super_glue = False ,save_model_path = '/tmp/',model_checkpoint = '../Code/bert2', use_prune_adapter = True):
    print(f"Start training {task}, with adapters {af_adapters}")
    print(f'Model saved in {save_model_path}')
    if use_prune_adapter:
        print("Using LTH model in AF")
        
    if not save_model_path.endswith('/'):
        save_model_path = save_model_path + '/'

    if is_super_glue:
        dataset_cache_dir = '/mnt/Code/super_glue/'+task
        metric_script = '/mnt/Code/util/super_glue.py'
    else:
        dataset_cache_dir = '/mnt/Code/dataset/'+task
        metric_script = '/mnt/Code/util/glue.py'

    model_checkpoint = '/mnt/Code/bert2'

    actual_task = "mnli" if task == "mnli-mm" else task

    print('Loading metric and datasets')

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
    # Encode the input data
    dataset = obtain_data.map(get_encode_batch(tokenizer, sentence1_key, sentence2_key), batched=True)
    print('Dataset is ready')

    # # Load from saved adapter
    print('Load from saved adapter')


    model_checkpoint = '../Code/bert2'
    
    num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2

    model = AutoModelWithHeads.from_pretrained(model_checkpoint)
    save_adapters_path = 'adapters_for_af'if use_prune_adapter else 'adapters_for_af_base'
    load_adapters(save_adapters_path, af_adapters, model, drop_layer = use_prune_adapter)

    # Add a fusion layer for all loaded adapters
    model.add_adapter_fusion(Fuse(*af_adapters))
    # Add a classification head for our target task
    model.add_classification_head(task, num_labels=num_labels)



    adapter_setup = Fuse(*af_adapters)
    model.train_adapter_fusion(adapter_setup)
    model.set_active_adapters(Fuse(*af_adapters))

    print('Model finished setup')

    metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
    validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"

    training_args = TrainingArguments(
        learning_rate = 2e-5,
        num_train_epochs=4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        logging_steps=50,
        output_dir= save_model_path+task,
        overwrite_output_dir=True,
        remove_unused_columns=True,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        metric_for_best_model=metric_name,
        load_best_model_at_end=True,
    )



    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = dataset['train'],
        eval_dataset = dataset[validation_key],
        tokenizer = tokenizer,
        compute_metrics = get_compute_metrics(metric, task),

    )

    print('Constructed trainer, start training')

    trainer.train()

    print('Training finished, Adding forward hooks to check connections')
    connections = defaultdict(dict)
    imp_dict = defaultdict(dict)



    handles = []
    for i in range(12):
        handle_att = add_register(trainer.model.bert.encoder.layer[i].attention.output,'attention', i, connections, imp_dict)
        handle_out = add_register(trainer.model.bert.encoder.layer[i].output,'output', i, connections, imp_dict)
        handles.extend(handle_att)
        handles.extend(handle_out)


    eval_result = trainer.evaluate()
    print(eval_result)
    # Write eval_result to file
    with open(save_model_path+task+'/eval_result.pkl', 'wb') as f:
        pickle.dump(eval_result, f)


    print("Finished checking")
    total_adapters = 0
    used_adapters = 0
    for k,v in imp_dict.items():
        print(k)
        print('Ori\t',end='')
        for name in v['adapter_names']:
            print(name, end='\t')
        total_adapters += len(v['adapter_names'])

        print()
        scores = v['importance'].mean(dim=-1).numpy()
        for i in range(len(scores)):
            score = scores[i]
            print(f"{score:.2f}", end='\t')
            if score >= 0.01 and i!=0:
                used_adapters += 1
        print()

    if total_adapters!=0:
        print("Total adapters",total_adapters, "Used adapters",used_adapters, "Used percent",used_adapters/ total_adapters)



    # Write importance to file
    with open(save_model_path+task+'/importance.pkl', 'wb') as f:
        pickle.dump(imp_dict, f)

    print('Finished evaluation, removing hooks')
    for handle in handles:
        handle.remove()

    print('hooks removed')



    


if __name__ == '__main__':
    # af_adapters = ["mnli", "qqp",'mrpc', 'rte', 'sst2', 'qnli']
    # tasks = ['mrpc', 'rte', 'sst2',  'qnli','qqp', 'mnli']
    af_adapters = ["mnli", "qqp",'mrpc', 'qnli','rte', 'sst2']
    tasks = ['mrpc','rte', 'sst2']

    for task in tasks:
        test(task, af_adapters, save_model_path='/tmp/', use_prune_adapter=True)