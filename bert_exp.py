import torch
from torch.optim import AdamW

import wandb
import math
import re
import warnings
from tqdm.auto import tqdm
import config as cfg
import argparse
from copy import deepcopy
from composer.core import Algorithm, Event
from composer import Trainer
from composer.optim import DecoupledAdamW, CosineAnnealingScheduler
from composer.models.huggingface import HuggingFaceModel
from composer.callbacks import LRMonitor, OptimizerMonitor, NaNMonitor
# from composer.loggers import WandBLogger

from utils.PyTransformer.transformers.torchTransformer import TorchTransformer
from utils.bert_pruner_quantizer import BERT_PRUNER

from transformers import default_data_collator
from transformers.models.bert.modeling_bert import BertSelfAttention, BertSelfOutput
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import get_scheduler, EvalPrediction
import evaluate
from accelerate import Accelerator

from QuantAttention import CustomizeBertSelfAttention, CustomizeBertSelfOutput
from datasets import load_dataset
from bert_utils import *
from bert_watch import EpochMonitor
from qa_utils import postprocess_qa_predictions

from torch.utils.data import DataLoader


def replace_attn_layer(module, config):
    if isinstance(module, BertSelfAttention):
        target_state_dict   = deepcopy(module.state_dict())
        new_module          = CustomizeBertSelfAttention(config, position_embedding_type=module.position_embedding_type)
        new_module.load_state_dict(target_state_dict)
        return new_module
    elif isinstance(module, BertSelfOutput):
        target_state_dict   = deepcopy(module.state_dict())
        new_module          = CustomizeBertSelfOutput(config)
        new_module.load_state_dict(target_state_dict)
        return new_module
    else:
        return module
    
def recursive_setattr(obj, attr, value):
    attr = attr.split('.', 1)
    if len(attr) == 1:
        setattr(obj, attr[0], value)
    else:
        recursive_setattr(getattr(obj, attr[0]), attr[1], value)



def watch_quantize_weight(model):
    
    for name, m in model.named_modules():
        if isinstance(m, CustomizeBertSelfAttention):
            query_mu, key_mu,value_mu = m.getMu()
            querySweight, keySweight, valueSweight = m.get_Sweight()
            queryPweight, keyPweight, valuePweight = m.get_Pweight()

            wandb.log({name+"_S_query": wandb.Histogram(querySweight.data.cpu().numpy())}, commit=False)
            wandb.log({name+"_S_key": wandb.Histogram(keySweight.data.cpu().numpy())}, commit=False)
            wandb.log({name+"_S_value": wandb.Histogram(valueSweight.data.cpu().numpy())}, commit=False)


            wandb.log({name+"_Mu_query": wandb.Histogram(query_mu)}, commit=False)
            wandb.log({name+"_Mu_key": wandb.Histogram(key_mu)}, commit=False)
            wandb.log({name+"_Mu_value": wandb.Histogram(value_mu)}, commit=False)


# def watch_lr(optimizer):


def main():
    parser = argparse.ArgumentParser(description="Unify Pruning and Quantization via VI on BERT model",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=512,
                        help="Seed Number.")
    parser.add_argument('--save_folder', type=str,
                        help="Output dir.")
    parser.add_argument('--run_name', type=str, default="BERT Quantization and Pruning.",
                        help="Run name")
    parser.add_argument('--project_name', type=str,
                        help='Projct name used for wandb.')
    parser.add_argument('--watch', action='store_true', default=False,
                        help='Whether choose to watch the paremters of model.')
    parser.add_argument('--dataset_name', type=str, default=None,
                        help='Dataset')
    parser.add_argument('--pad_to_max_length', action="store_true",
                        help="If passed, pad all samples to `max_seq_length`. Otherwise, dynamic padding is used.")
    parser.add_argument('--autoresume', action='store_true', default=False,
                        help="Auto Resume the training process.")
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument("--overwrite_cache", action="store_true", 
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--train_file", type=str, default=None, 
                        help="A csv or a json file containing the training data.")
    parser.add_argument("--validation_file", type=str, default=None, 
                        help="A csv or a json file containing the validation data.")
    parser.add_argument("--test_file", type=str, default=None, 
                        help="A csv or a json file containing the Prediction data.")
    parser.add_argument('--empirical', type=bool, default=False,
                        help='whether use empirical initialization for parameter sigma (default: False)')
    parser.add_argument('--normal', action='store_true', default=False,
                        help='whether train noramlly (default: False)')
    parser.add_argument('--K', type=int, default=4, metavar='K',
                    help='number of GMM components (default: 2^4=16)')
    parser.add_argument('--tau', type=float, default=0.01, metavar='TAU',
                        help='gumbel softmax temperature (default: 0.01)')
    parser.add_argument('--init_method', type=str, default='k-means',
                        choices=['k-means', 'quantile', 'empirical'],
                        help='Choose Initialization Method for Mu')
    parser.add_argument('--prune', action='store_true', default=False,
                        help="Prune or Not")
    parser.add_argument('--prune_scale', type=float, default=0.2,
                        help='Scale the pruning parameter by 1/prune_scale')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='When debug skip initialization')
    parser.add_argument('--freeze_weight', action='store_true', default=False,
                        help='Freeze Parameters')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch Size')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help="Initial Learning rate.")
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--duration', type=int, default=20,
                        help="Number of Epochs")
    parser.add_argument('--warm_up', type=str, default='2ep',
                        help='Warm Up epoch before pruning')
    parser.add_argument('--init_sparsity', type=float, default=0.0,
                        help='Begin with this intial step size.')
    parser.add_argument('--final_sparsity', type=float, default=0.7,
                        help='The target sparsity.')
    parser.add_argument('--pr_warm_up', type=str, default='2ep',
                        help='Warm Up epoch before pruning')
    parser.add_argument('--sigma', type=float, default=3,
                        help="Initial Sigma for the prior distribution.")
    parser.add_argument('--max_length', type=int, default=384,
                        help="max length of LLM model features")
    parser.add_argument('--doc_stride', type=int, default=128,
                        help='stride length for features')
    parser.add_argument('--alpha_f', type=float, default=0.01,
                        help='The ratio between the final LR and initial LR.')
    parser.add_argument('--eval_interval', type=str, default='5ep',
                        help='Frequence of validation.')
    parser.add_argument('--prune_end', type=float, default=3,
                        help='End of the Pruning')
    parser.add_argument('--prune_start', type=float, default=1, 
                        help='Starting point of the pruning process.')
    parser.add_argument('--sample', action='store_true', default=False,
                        help="Whether sample quantization weights or take the maximum")
    parser.add_argument('--average', action='store_true', default=False,
                        help="Whether use Bayesian Average to ensemble model.")
    
    
    args = parser.parse_args()
    

    cfg.set_config(args=args)

    wandb.login()

    wandb.init(project='Quantization_BERT_SQUAD', name=args.run_name)

    # Load the pretrained model properly. 
    tokenizer = AutoTokenizer.from_pretrained("huggingface-course/bert-finetuned-squad")
    model = AutoModelForQuestionAnswering.from_pretrained("huggingface-course/bert-finetuned-squad")

    config = model.config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    max_length = min(args.max_length, tokenizer.model_max_length)
    doc_stride = args.doc_stride


    
    pad_on_right = tokenizer.padding_side == "right"
        # Training preprocessing
    def prepare_train_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples["question"] = [q.lstrip() for q in examples["question"]]

        # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples["question" if pad_on_right else "context"],
            examples["context" if pad_on_right else "question"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples
    
    def prepare_validation_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples["question"] = [q.lstrip() for q in examples["question"]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples["question" if pad_on_right else "context"],
            examples["context" if pad_on_right else "question"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # We keep the example_id that gave us this feature and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples
    

    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.

        ###################################
        #  Use cached dataset if possible #
        ###################################
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name, cache_dir="./cache")
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
            extension = args.train_file.split(".")[-1]
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
            extension = args.validation_file.split(".")[-1]
        if args.test_file is not None:
            data_files["test"] = args.test_file
            extension = args.test_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, field="data")

    column_names = raw_datasets["train"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]


    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=False,
            n_best_size=n_best,
            max_answer_length=max_answer_length,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)
    


    tokenized_train_data = raw_datasets['train'].map(prepare_train_features, 
                                                    batched=True, 
                                                    remove_columns=raw_datasets['train'].column_names,
                                                    )

    tokenized_train_data.set_format('torch')

    tokenized_validation_data = raw_datasets["validation"].map(
                                                        prepare_validation_features,
                                                        batched=True,
                                                        remove_columns=raw_datasets["validation"].column_names,
                                                        )
    
    validation_data_for_model=tokenized_validation_data.remove_columns(["example_id", "offset_mapping"])
    validation_data_for_model.set_format('torch')
    

    train_dataloader = DataLoader(
        tokenized_train_data,
        shuffle=True,
        collate_fn=customized_data_collator,
        batch_size=12,
    )

    eval_dataloader = DataLoader(
        validation_data_for_model,
        collate_fn=customized_valid_data_collator,
        batch_size=args.batch_size,
    )



    metric = evaluate.load('squad')

    num_train_epochs = args.duration
    num_update_steps_per_epoch = len(train_dataloader)
    cfg.TOT_TRAIN_STEP = num_training_steps = num_train_epochs * num_update_steps_per_epoch
    cfg.PRUNE_END_STEP = len(train_dataloader)*args.prune_end
    cfg.PRUNE_START_STEP = len(train_dataloader)*args.prune_start




    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        eps=1e-8
    )

    accelerator = Accelerator(mixed_precision='fp16')
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    pruner = BERT_PRUNER(model, init_sparsity=args.init_sparsity , final_sparsity=args.final_sparsity, alpha_f=0.1)

    lr_scheduler = get_scheduler(
        'cosine',
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=cfg.PRUNE_END_STEP,
    )


    # Report the unquantized full-precision model
    model.eval()
    cfg.IS_TRAIN = False
    all_start_logits = []
    all_end_logits = []
    accelerator.print("Evaluation!")
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        start_logits = outputs.start_logits
        end_logits = outputs.end_logits


        if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
            start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
            end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)

        all_start_logits.append(accelerator.gather_for_metrics(start_logits).cpu().numpy())
        all_end_logits.append(accelerator.gather_for_metrics(end_logits).cpu().numpy())

    max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor

    # concatenate the numpy array
    start_logits_concat = create_and_fill_np_array(all_start_logits, tokenized_validation_data, max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, tokenized_validation_data, max_len)

    # delete the list of numpy arrays
    del all_start_logits
    del all_end_logits

    outputs_numpy = (start_logits_concat, end_logits_concat)
    prediction = post_processing_function(raw_datasets['validation'], tokenized_validation_data, outputs_numpy)
    eval_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
    # wandb.log({f"Evaluation metrics:" eval_metric}, commit=False)

    for key in eval_metric.keys():
        wandb.log({'Oringinal Validation '+str(key): eval_metric[key]}, commit=False)

    accelerator.wait_for_everyone()

    for name, module in tuple(model.named_modules()):
        if name:
            recursive_setattr(model, name, replace_attn_layer(module, config))

    InitBertModel(model, args.sigma)

    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_train_epochs):
    # Training
        if args.watch:
            watch_quantize_weight(model)
        model.train()
        wandb.log({'epoch': epoch}, commit=False)
        cfg.IS_TRAIN=True
        for step, batch in enumerate(train_dataloader):
            curr_step = len(train_dataloader)*epoch+step
            pruner.prune(curr_step)
            pruner.log_sparsity()
            pruner.monitor_scheduler_step(optimizer)

            outputs = model(**batch)
            loss = outputs.loss
            wandb.log({'Training Loss': loss})
            accelerator.backward(loss)
            pruner.apply_non_prune_gradient(step)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # Evaluation
        model.eval()
        cfg.IS_TRAIN = False
        all_start_logits = []
        all_end_logits = []
        accelerator.print("Evaluation!")
        for batch in tqdm(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            start_logits = outputs.start_logits
            end_logits = outputs.end_logits


            if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
                end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)

            all_start_logits.append(accelerator.gather_for_metrics(start_logits).cpu().numpy())
            all_end_logits.append(accelerator.gather_for_metrics(end_logits).cpu().numpy())

        max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor

        # concatenate the numpy array
        start_logits_concat = create_and_fill_np_array(all_start_logits, tokenized_validation_data, max_len)
        end_logits_concat = create_and_fill_np_array(all_end_logits, tokenized_validation_data, max_len)

        # delete the list of numpy arrays
        del all_start_logits
        del all_end_logits

        outputs_numpy = (start_logits_concat, end_logits_concat)
        prediction = post_processing_function(raw_datasets['validation'], tokenized_validation_data, outputs_numpy)
        eval_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
        # wandb.log({f"Evaluation metrics:" eval_metric}, commit=False)

        for key in eval_metric.keys():
            wandb.log({'Validation '+str(key): eval_metric[key]}, commit=False)

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.save_folder, save_function=accelerator.save)



if __name__ == '__main__':

    main()
    