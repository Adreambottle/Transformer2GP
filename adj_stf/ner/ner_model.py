from __future__ import absolute_import, division, print_function

import json
import logging
import math
import os
import gc
import random
import warnings
from dataclasses import asdict
from multiprocessing import cpu_count
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm.auto import tqdm, trange
from adj_tf.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from adj_tf.optimization import AdamW, Adafactor
from adj_tf import (  # 载入 config_class, model_class, tokenizer_class
    AlbertConfig,
    AlbertForTokenClassification,
    AlbertTokenizer,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
    BertweetTokenizer,
    CamembertConfig,
    CamembertForTokenClassification,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForTokenClassification,
    DistilBertTokenizer,
    ElectraConfig,
    ElectraForTokenClassification,
    ElectraTokenizer,
    LayoutLMConfig,
    LayoutLMForTokenClassification,
    LayoutLMTokenizer,
    LongformerConfig,
    LongformerForTokenClassification,
    LongformerTokenizer,
    MPNetConfig,
    MPNetForTokenClassification,
    MPNetTokenizer,
    MobileBertConfig,
    MobileBertForTokenClassification,
    MobileBertTokenizer,
    RobertaConfig,
    RobertaForTokenClassification,
    RobertaTokenizer,
    SqueezeBertConfig,
    SqueezeBertForTokenClassification,
    SqueezeBertTokenizer,
    WEIGHTS_NAME,
    XLMRobertaConfig,
    XLMRobertaForTokenClassification,
    XLMRobertaTokenizer,
    XLNetConfig,
    XLNetForTokenClassification,
    XLNetTokenizerFast,
)
from wandb import config
from adj_tf.convert_graph_to_onnx import convert, quantize

from adj_stf.config.global_args import global_args
from adj_stf.config.model_args import NERArgs
from adj_stf.config.utils import sweep_config_to_sweep_values
from adj_stf.ner.ner_utils import (InputExample,
                                   convert_examples_to_features,
                                   get_examples_from_df,
                                   get_labels,
                                   read_examples_from_file,
                                   )

from adj_stf.ner.DNN2GP import (compute_dnn2gp_quantities,
                               compute_laplace,
                               token_length_and_first_token_id)

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False

logger = logging.getLogger(__name__)

MODELS_WITH_EXTRA_SEP_TOKEN = ["roberta", "camembert", "xlmroberta", "longformer", "mpnet"]


class NERModel:
    def __init__(
            self,
            model_type,  # model_type 是 eg "roberta"
            model_name,  # model_name 是 eg "roberta-base" 也可以是自己的model
            labels=None,
            args=None,  # args 从配置文件中引入
            use_cuda=True,
            cuda_device=-1,
            onnx_execution_provider=None,
            **kwargs,
    ):
        """
        Initializes a NERModel

        Args:
            model_type: The type of model (bert, roberta)
            model_name: Default Transformer model name or path to a directory containing Transformer model file (pytorch_model.bin).
            labels (optional): A list of all Named Entity labels.  If not given, ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"] will be used.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"

        MODEL_CLASSES = {  # config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
            "albert": (AlbertConfig, AlbertForTokenClassification, AlbertTokenizer),
            "auto": (AutoConfig, AutoModelForTokenClassification, AutoTokenizer),
            "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
            "bertweet": (RobertaConfig, RobertaForTokenClassification, BertweetTokenizer),
            "camembert": (CamembertConfig, CamembertForTokenClassification, CamembertTokenizer),
            "distilbert": (DistilBertConfig, DistilBertForTokenClassification, DistilBertTokenizer),
            "electra": (ElectraConfig, ElectraForTokenClassification, ElectraTokenizer),
            "layoutlm": (LayoutLMConfig, LayoutLMForTokenClassification, LayoutLMTokenizer),
            "longformer": (LongformerConfig, LongformerForTokenClassification, LongformerTokenizer),
            "mobilebert": (MobileBertConfig, MobileBertForTokenClassification, MobileBertTokenizer),
            "mpnet": (MPNetConfig, MPNetForTokenClassification, MPNetTokenizer),
            "roberta": (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer),
            "squeezebert": (SqueezeBertConfig, SqueezeBertForTokenClassification, SqueezeBertTokenizer),
            "xlmroberta": (XLMRobertaConfig, XLMRobertaForTokenClassification, XLMRobertaTokenizer),
            "xlnet": (XLNetConfig, XLNetForTokenClassification, XLNetTokenizerFast),
        }  # Key 就是 model_type

        self.args = self._load_model_args(model_name)  # 按照 model_name 载入该模型需要的 args

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, NERArgs):
            self.args = args

        if "sweep_config" in kwargs:
            self.is_sweeping = True
            sweep_config = kwargs.pop("sweep_config")
            sweep_values = sweep_config_to_sweep_values(sweep_config)
            self.args.update_from_dict(sweep_values)
        else:
            self.is_sweeping = False

        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if self.args.n_gpu > 0:
                torch.cuda.manual_seed_all(self.args.manual_seed)

        if not use_cuda:  # 如果不能用 cuda 不能用 fp16
            self.args.fp16 = False

        if labels and self.args.labels_list:  # 载入输入的 labels
            assert labels == self.args.labels_list
            self.args.labels_list = labels
        elif labels:
            self.args.labels_list = labels
        elif self.args.labels_list:
            pass
        else:
            self.args.labels_list = [
                "O",
                "B-MISC",
                "I-MISC",
                "B-PER",
                "I-PER",
                "B-ORG",
                "I-ORG",
                "B-LOC",
                "I-LOC",
            ]
        self.num_labels = len(self.args.labels_list)  # labels 的长度
        # 定义来自 transformer 的 config_class, model_class, tokenizer_class
        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        if self.num_labels:
            self.config = config_class.from_pretrained(model_name, num_labels=self.num_labels, **self.args.config)
            self.num_labels = self.num_labels
        else:
            self.config = config_class.from_pretrained(model_name, **self.args.config)
            self.num_labels = self.config.num_labels

        if use_cuda:  # use_cuda 是否使用cuda，来自用户输入
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    "Make sure CUDA is available or set use_cuda=False."
                )
        else:
            self.device = "cpu"

        if self.args.onnx:  # 是否使用用 oxxn
            from onnxruntime import InferenceSession, SessionOptions

            if not onnx_execution_provider:
                onnx_execution_provider = "CUDAExecutionProvider" if use_cuda else "CPUExecutionProvider"

            options = SessionOptions()
            options.intra_op_num_threads = 1

            if self.args.dynamic_quantize:
                model_path = quantize(Path(os.path.join(model_name, "onnx_model.onnx")))
                self.model = InferenceSession(model_path.as_posix(), options, providers=[onnx_execution_provider])
            else:
                model_path = os.path.join(model_name, "onnx_model.onnx")
                self.model = InferenceSession(model_path, options, providers=[onnx_execution_provider])
        else:
            if not self.args.quantized_model:
                self.model = model_class.from_pretrained(model_name, config=self.config, **kwargs)  # self.model 的定义部分
            else:
                quantized_weights = torch.load(os.path.join(model_name, "pytorch_model.bin"))  # 载入自己的 model 模型
                self.model = model_class.from_pretrained(None, config=self.config, state_dict=quantized_weights)

            if self.args.dynamic_quantize:
                self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
            if self.args.quantized_model:
                self.model.load_state_dict(quantized_weights)
            if self.args.dynamic_quantize:
                self.args.quantized_model = True

        self.results = {}

        if self.args.fp16:  # 使用 fp16 的格式
            try:
                from torch.cuda import amp
            except AttributeError:
                raise AttributeError("fp16 requires Pytorch >= 1.6. Please update Pytorch or turn off fp16.")

        if model_name in [  # bertweet 是需要 normalization 的
            "vinai/bertweet-base",
            "vinai/bertweet-covid19-base-cased",
            "vinai/bertweet-covid19-base-uncased",
        ]:
            self.tokenizer = tokenizer_class.from_pretrained(
                model_name, do_lower_case=self.args.do_lower_case, normalization=True, **kwargs
            )
        else:
            self.tokenizer = tokenizer_class.from_pretrained(  # 在这里用 from_pretrained
                model_name, do_lower_case=self.args.do_lower_case, **kwargs
            )

        if self.args.special_tokens_list:
            self.tokenizer.add_tokens(self.args.special_tokens_list, special_tokens=True)
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.args.model_name = model_name  # from input
        self.args.model_type = model_type  # from input

        self.pad_token_label_id = CrossEntropyLoss().ignore_index  # pad_token_label_id 这个是交叉熵 activate function

        if model_type == "camembert":
            warnings.warn(
                "use_multiprocessing automatically disabled as CamemBERT"
                " fails when using multiprocessing for feature conversion."
            )
            self.args.use_multiprocessing = False

        if self.args.wandb_project and not wandb_available:
            warnings.warn("wandb_project specified but wandb is not available. Wandb disabled.")
            self.args.wandb_project = None

    def train_model(
            self, train_data, output_dir=None, show_running_loss=True, args=None, eval_data=None, verbose=True, **kwargs
    ):
        """
        Trains the model using 'train_data'

        Args:
            train_data: train_data should be the path to a .txt file containing the training data OR a pandas DataFrame with 3 columns.
                        If a text file is given the data should be in the CoNLL format. i.e. One word per line, with sentences seperated by an empty line.
                        The first word of the line should be a word, and the last should be a Name Entity Tag.
                        If a DataFrame is given, each sentence should be split into words, with each word assigned a tag, and with all words from the same sentence given the same sentence_id.
            eval_data: Evaluation data (same format as train_data) against which evaluation will be performed when evaluate_during_training is enabled. Is required if evaluate_during_training is enabled.
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            show_running_loss (optional): Set to False to prevent running loss from being printed to console. Defaults to True.
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

        Returns:
            global_step: Number of global steps trained
            training_details: Average training loss if evaluate_during_training is False or full training progress scores if evaluate_during_training is True
        """  # noqa: ignore flake8"

        if args:
            self.args.update_from_dict(args)  # 从 args 中导入

        if self.args.silent:  # 是否展示loss
            show_running_loss = False

        if self.args.evaluate_during_training and eval_data is None:
            if "eval_df" in kwargs:
                warnings.warn(
                    "The eval_df parameter has been renamed to eval_data."
                    " Using eval_df will raise an error in a future version."
                )
                eval_data = kwargs.pop("eval_df")
            else:
                raise ValueError(
                    "evaluate_during_training is enabled but eval_data is not specified."
                    " Pass eval_data to model.train_model() if using evaluate_during_training."
                )

        if not output_dir:  # 将output返回的地址
            output_dir = self.args.output_dir

        if os.path.exists(output_dir) and os.listdir(output_dir) and not self.args.overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Use --overwrite_output_dir to overcome.".format(output_dir)
            )

        self._move_model_to_device()

        train_dataset = self.load_and_cache_examples(train_data)

        os.makedirs(output_dir, exist_ok=True)

        """开始训练模型"""
        global_step, training_details = self.train(  # 这里开始train这个模型
            train_dataset, output_dir, show_running_loss=show_running_loss, eval_data=eval_data, **kwargs
        )  # 这两个是 train 函数的返回值

        self.save_model(model=self.model)

        logger.info(" Training of {} model complete. Saved to {}.".format(self.args.model_type, output_dir))

        return global_step, training_details

    def train(self, train_dataset, output_dir, show_running_loss=True, eval_data=None, verbose=True, **kwargs):
        """
        Trains the model on train_dataset.

        Utility function to be used by the train_model() method. Not intended to be used directly.
        """

        device = self.device
        model = self.model  # model 是来自 self.model 里面的
        print(model)
        args = self.args  # args 是来自 self.args

        tb_writer = SummaryWriter(logdir=args.tensorboard_dir)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(  # 这里是 DataLoader(train_dataset, sampler, batch_size, num_worker)
            train_dataset,
            sampler=train_sampler,
            batch_size=args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
        )

        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = []  # optimizer_grouped_parameters 分组的方式来储存参数
        custom_parameter_names = set()
        for group in self.args.custom_parameter_groups:  # group 是参数的集合， custom_parameter_groups 是个list
            params = group.pop("params")
            custom_parameter_names.update(params)
            param_group = {**group}
            param_group["params"] = [p for n, p in model.named_parameters() if n in params]
            optimizer_grouped_parameters.append(param_group)

        for group in self.args.custom_layer_parameters:
            layer_number = group.pop("layer")
            layer = f"layer.{layer_number}."
            group_d = {**group}
            group_nd = {**group}
            group_nd["weight_decay"] = 0.0
            params_d = []
            params_nd = []
            for n, p in model.named_parameters():
                if n not in custom_parameter_names and layer in n:
                    if any(nd in n for nd in no_decay):
                        params_nd.append(p)
                    else:
                        params_d.append(p)
                    custom_parameter_names.add(n)
            group_d["params"] = params_d
            group_nd["params"] = params_nd

            optimizer_grouped_parameters.append(group_d)
            optimizer_grouped_parameters.append(group_nd)

        if not self.args.train_custom_parameters_only:
            optimizer_grouped_parameters.extend(
                [
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names and not any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names and any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            )

        warmup_steps = math.ceil(t_total * args.warmup_ratio)
        args.warmup_steps = warmup_steps if args.warmup_steps == 0 else args.warmup_steps

        if args.optimizer == "AdamW":
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        elif args.optimizer == "Adafactor":
            optimizer = Adafactor(  # 定义优化器
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                eps=args.adafactor_eps,
                clip_threshold=args.adafactor_clip_threshold,
                decay_rate=args.adafactor_decay_rate,
                beta1=args.adafactor_beta1,
                weight_decay=args.weight_decay,
                scale_parameter=args.adafactor_scale_parameter,
                relative_step=args.adafactor_relative_step,
                warmup_init=args.adafactor_warmup_init,
            )
            print("Using Adafactor for T5")
        else:
            raise ValueError(
                "{} is not a valid optimizer class. Please use one of ('AdamW', 'Adafactor') instead.".format(
                    args.optimizer
                )
            )

        if args.scheduler == "constant_schedule":
            scheduler = get_constant_schedule(optimizer)

        elif args.scheduler == "constant_schedule_with_warmup":
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)

        elif args.scheduler == "linear_schedule_with_warmup":
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
            )

        elif args.scheduler == "cosine_schedule_with_warmup":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                num_cycles=args.cosine_schedule_num_cycles,
            )

        elif args.scheduler == "cosine_with_hard_restarts_schedule_with_warmup":
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                num_cycles=args.cosine_schedule_num_cycles,
            )

        elif args.scheduler == "polynomial_decay_schedule_with_warmup":
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                lr_end=args.polynomial_decay_schedule_lr_end,
                power=args.polynomial_decay_schedule_power,
            )

        else:
            raise ValueError("{} is not a valid scheduler.".format(args.scheduler))

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)  # 如果是多个GPU的形式，采用并行计算的模式

        global_step = 0  # 定义超参数
        training_progress_scores = None
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()  # 将model积累的gradient清零
        train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.silent, mininterval=0)  # 这是个迭代器
        epoch_number = 0
        best_eval_metric = None
        early_stopping_counter = 0
        steps_trained_in_current_epoch = 0
        epochs_trained = 0

        if args.model_name and os.path.exists(args.model_name):  # 判断是否存在 model_name 和 路径下存在 model 的文件
            try:
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = args.model_name.split("/")[-1].split("-")  # 这是个后缀
                if len(checkpoint_suffix) > 2:
                    checkpoint_suffix = checkpoint_suffix[1]
                else:
                    checkpoint_suffix = checkpoint_suffix[-1]
                global_step = int(checkpoint_suffix)  # global_step 是一共要走多少个 step
                epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = global_step % (
                        len(train_dataloader) // args.gradient_accumulation_steps
                )  # 这是第几个 batch 的意思吗

                logger.info("   Continuing training from checkpoint, will skip to saved global_step")
                logger.info("   Continuing training from epoch %d", epochs_trained)
                logger.info("   Continuing training from global step %d", global_step)
                logger.info("   Will skip the first %d steps in the current epoch", steps_trained_in_current_epoch)
            except ValueError:
                logger.info("   Starting fine-tuning.")

        if args.evaluate_during_training:
            training_progress_scores = self._create_training_progress_scores(**kwargs)
        if args.wandb_project:
            wandb.init(project=args.wandb_project, config={**asdict(args)}, **args.wandb_kwargs)
            wandb.watch(self.model)

        if self.args.fp16:
            from torch.cuda import amp

            scaler = amp.GradScaler()

        for _ in train_iterator:  # 这里开始训练模型，train_iterator 迭代的是Epoch
            model.train()  # 开始 train()
            if epochs_trained > 0:  # 如果 epochs_trained > 0 跳出本次循环
                epochs_trained -= 1
                continue
            train_iterator.set_description(f"Epoch {epoch_number + 1} of {args.num_train_epochs}")  # 设置进度条的输出
            batch_iterator = tqdm(  # batch_iterator 是 train_dataloader
                train_dataloader,  # 每个Epoch内全部的 train_data 进行 DataLoader 的封装处理
                desc=f"Running Epoch {epoch_number} of {args.num_train_epochs}",
                disable=args.silent,
                mininterval=0,
            )
            for step, batch in enumerate(batch_iterator):  # 这里开始对每个batch进行循环，step 是第几个 batch
                if steps_trained_in_current_epoch > 0:  # 第几个 step
                    steps_trained_in_current_epoch -= 1
                    continue
                batch = tuple(t.to(device) for t in batch)  # 用tuple储存分配在device上的batch里的数据

                inputs = self._get_inputs_dict(batch)  # 返回 inputs 这个dictionary，放在 model() 的输入里

                if self.args.fp16:  # 如果是 fp16 的格式
                    with amp.autocast():
                        outputs = model(**inputs)
                        # model outputs are always tuple in pytorch-adj_tf (see doc)
                        loss = outputs[0]
                else:
                    outputs = model(**inputs)  # 不是 fp16 的格式，直接把 inputs 作为输入放入model中
                    # model outputs are always tuple in pytorch-adj_tf (see doc)
                    loss = outputs[0]  # loss 是outputs的第0个返回值

                if args.n_gpu > 1:  # 如果多个gpu计算loss的mean
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training

                current_loss = loss.item()  # 记录当前的 loss 值

                if show_running_loss:  # 展示当前loss
                    batch_iterator.set_description(
                        f"Epochs {epoch_number}/{args.num_train_epochs}. Running Loss: {current_loss:9.4f}"
                    )

                if args.gradient_accumulation_steps > 1:  # 有的加和loss的是大于1的
                    loss = loss / args.gradient_accumulation_steps

                if self.args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()  # 这里开始回传 bp

                tr_loss += loss.item()  # 用loss的数值加和，tr_loss 是全部的loss
                if (step + 1) % args.gradient_accumulation_steps == 0:  # 每隔一个step加一回loss
                    if self.args.fp16:
                        scaler.unscale_(optimizer)
                    if args.optimizer == "AdamW":
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    if self.args.fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()  # 在这里清零 model 里的
                    global_step += 1  # global_step

                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        # Log metrics                         # 这部分是记录log用的
                        tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                        tb_writer.add_scalar(
                            "loss", (tr_loss - logging_loss) / args.logging_steps, global_step,
                        )
                        logging_loss = tr_loss  # log里的loss值
                        if args.wandb_project or self.is_sweeping:
                            wandb.log(  # log召开那个输出的成分
                                {
                                    "Training loss": current_loss,
                                    "lr": scheduler.get_last_lr()[0],
                                    "global_step": global_step,
                                }
                            )

                    if args.save_steps > 0 and global_step % args.save_steps == 0:  # 每隔几个Step保存一次模型
                        # Save model checkpoint               # 保存模型的 ckeckpoint
                        output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))

                        self.save_model(output_dir_current, optimizer, scheduler, model=model)

                    if args.evaluate_during_training and (  # 如果在训练的时候就要给出 evaluate
                            args.evaluate_during_training_steps > 0
                            and global_step % args.evaluate_during_training_steps == 0
                    ):
                        # 保存输出的结果
                        output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))

                        os.makedirs(output_dir_current, exist_ok=True)

                        # Only evaluate when single GPU otherwise metrics may not average well
                        results, _, _ = self.eval_model(
                            eval_data,
                            verbose=verbose and args.evaluate_during_training_verbose,
                            wandb_log=False,
                            output_dir=output_dir_current,
                            **kwargs,
                        )
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)

                        if args.save_eval_checkpoints:
                            self.save_model(output_dir_current, optimizer, scheduler, model=model, results=results)

                        training_progress_scores["global_step"].append(global_step)
                        training_progress_scores["train_loss"].append(current_loss)
                        for key in results:
                            training_progress_scores[key].append(results[key])
                        report = pd.DataFrame(training_progress_scores)
                        report.to_csv(
                            os.path.join(args.output_dir, "training_progress_scores.csv"), index=False,
                        )

                        if args.wandb_project or self.is_sweeping:
                            wandb.log(self._get_last_metrics(training_progress_scores))

                        if not best_eval_metric:
                            best_eval_metric = results[args.early_stopping_metric]
                            self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                        if best_eval_metric and args.early_stopping_metric_minimize:
                            if results[args.early_stopping_metric] - best_eval_metric < args.early_stopping_delta:
                                best_eval_metric = results[args.early_stopping_metric]
                                self.save_model(
                                    args.best_model_dir, optimizer, scheduler, model=model, results=results
                                )
                                early_stopping_counter = 0
                            else:
                                if args.use_early_stopping:
                                    if early_stopping_counter < args.early_stopping_patience:
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(f" No improvement in {args.early_stopping_metric}")
                                            logger.info(f" Current step: {early_stopping_counter}")
                                            logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                                    else:
                                        if verbose:
                                            logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return (
                                            global_step,
                                            tr_loss / global_step
                                            if not self.args.evaluate_during_training
                                            else training_progress_scores,
                                        )
                        else:
                            if results[args.early_stopping_metric] - best_eval_metric > args.early_stopping_delta:
                                best_eval_metric = results[args.early_stopping_metric]
                                self.save_model(
                                    args.best_model_dir, optimizer, scheduler, model=model, results=results
                                )
                                early_stopping_counter = 0
                            else:
                                if args.use_early_stopping:
                                    if early_stopping_counter < args.early_stopping_patience:
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(f" No improvement in {args.early_stopping_metric}")
                                            logger.info(f" Current step: {early_stopping_counter}")
                                            logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                                    else:
                                        if verbose:
                                            logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return (
                                            global_step,
                                            tr_loss / global_step
                                            if not self.args.evaluate_during_training
                                            else training_progress_scores,
                                        )
                        model.train()

            epoch_number += 1  # 保存输出的结果，记录路径
            output_dir_current = os.path.join(output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch_number))

            if args.save_model_every_epoch or args.evaluate_during_training:  # 如果训练中保存
                os.makedirs(output_dir_current, exist_ok=True)

            if args.save_model_every_epoch:
                self.save_model(output_dir_current, optimizer, scheduler, model=model)

            if args.evaluate_during_training and args.evaluate_each_epoch:  # 每个epoch都要保存
                results, _, _ = self.eval_model(
                    eval_data, verbose=verbose and args.evaluate_during_training_verbose, wandb_log=False, **kwargs
                )

                self.save_model(output_dir_current, optimizer, scheduler, results=results)

                training_progress_scores["global_step"].append(global_step)
                training_progress_scores["train_loss"].append(current_loss)
                for key in results:
                    training_progress_scores[key].append(results[key])
                report = pd.DataFrame(training_progress_scores)
                report.to_csv(os.path.join(args.output_dir, "training_progress_scores.csv"), index=False)

                if args.wandb_project or self.is_sweeping:
                    wandb.log(self._get_last_metrics(training_progress_scores))

                if not best_eval_metric:
                    best_eval_metric = results[args.early_stopping_metric]
                    self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                if best_eval_metric and args.early_stopping_metric_minimize:
                    if results[args.early_stopping_metric] - best_eval_metric < args.early_stopping_delta:
                        best_eval_metric = results[args.early_stopping_metric]
                        self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                        early_stopping_counter = 0
                    else:
                        if args.use_early_stopping and args.early_stopping_consider_epochs:
                            if early_stopping_counter < args.early_stopping_patience:
                                early_stopping_counter += 1
                                if verbose:
                                    logger.info(f" No improvement in {args.early_stopping_metric}")
                                    logger.info(f" Current step: {early_stopping_counter}")
                                    logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                            else:
                                if verbose:
                                    logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                    logger.info(" Training terminated.")
                                    train_iterator.close()
                                return (
                                    global_step,
                                    tr_loss / global_step
                                    if not self.args.evaluate_during_training
                                    else training_progress_scores,
                                )
                else:
                    if results[args.early_stopping_metric] - best_eval_metric > args.early_stopping_delta:
                        best_eval_metric = results[args.early_stopping_metric]
                        self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                        early_stopping_counter = 0
                        early_stopping_counter = 0
                    else:
                        if args.use_early_stopping and args.early_stopping_consider_epochs:
                            if early_stopping_counter < args.early_stopping_patience:
                                early_stopping_counter += 1
                                if verbose:
                                    logger.info(f" No improvement in {args.early_stopping_metric}")
                                    logger.info(f" Current step: {early_stopping_counter}")
                                    logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                            else:
                                if verbose:
                                    logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                    logger.info(" Training terminated.")
                                    train_iterator.close()
                                return (
                                    global_step,
                                    tr_loss / global_step
                                    if not self.args.evaluate_during_training
                                    else training_progress_scores,
                                )

        return (  # train 函数的返回loss
            global_step,
            tr_loss / global_step if not self.args.evaluate_during_training else training_progress_scores,
        )

    def eval_model(self, eval_data, output_dir=None, verbose=True, silent=False, wandb_log=True, **kwargs):
        """
        Evaluates the model on eval_data. Saves results to output_dir.

        Args:
            eval_data: eval_data should be the path to a .txt file containing the evaluation data or a pandas DataFrame.
                        If a text file is used the data should be in the CoNLL format. I.e. One word per line, with sentences seperated by an empty line.
                        The first word of the line should be a word, and the last should be a Name Entity Tag.
                        If a DataFrame is given, each sentence should be split into words, with each word assigned a tag, and with all words from the same sentence given the same sentence_id.

            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            verbose: If verbose, results will be printed to the console on completion of evaluation.
            silent: If silent, tqdm progress bars will be hidden.
            wandb_log: If True, evaluation results will be logged to wandb.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

        Returns:
            result: Dictionary containing evaluation results. (eval_loss, precision, recall, f1_score)
            model_outputs: List of raw model outputs
            preds_list: List of predicted tags
        """  # noqa: ignore flake8"
        if not output_dir:
            output_dir = self.args.output_dir

        self._move_model_to_device()  # 调用 self.model.to(self.device)

        eval_dataset = self.load_and_cache_examples(eval_data, evaluate=True)

        result, model_outputs, preds_list = self.evaluate(
            eval_dataset, output_dir, verbose=verbose, silent=silent, wandb_log=wandb_log, **kwargs
        )
        self.results.update(result)

        if verbose:
            logger.info(self.results)

        return result, model_outputs, preds_list

    def evaluate(self, eval_dataset, output_dir, verbose=True, silent=False, wandb_log=True, **kwargs):
        """
        Evaluates the model on eval_dataset.

        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """

        device = self.device  # 设置CPU/GPU
        model = self.model  # get model
        args = self.args  # get args
        pad_token_label_id = self.pad_token_label_id  # get activate function
        eval_output_dir = output_dir

        results = {}

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        model.eval()

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if self.args.fp16:
            from torch.cuda import amp

        for batch in tqdm(eval_dataloader, disable=args.silent or silent, desc="Running Evaluation"):
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = self._get_inputs_dict(batch)

                if self.args.fp16:
                    with amp.autocast():
                        outputs = model(**inputs)
                        tmp_eval_loss, logits = outputs[:2]
                else:
                    outputs = model(**inputs)
                    tmp_eval_loss, logits = outputs[:2]

                if self.args.n_gpu > 1:
                    tmp_eval_loss = tmp_eval_loss.mean()
                eval_loss += tmp_eval_loss.item()

            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
                out_input_ids = inputs["input_ids"].detach().cpu().numpy()
                out_attention_mask = inputs["attention_mask"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
                out_input_ids = np.append(out_input_ids, inputs["input_ids"].detach().cpu().numpy(), axis=0)
                out_attention_mask = np.append(
                    out_attention_mask, inputs["attention_mask"].detach().cpu().numpy(), axis=0,
                )

        eval_loss = eval_loss / nb_eval_steps
        token_logits = preds
        preds = np.argmax(preds, axis=2)

        label_map = {i: label for i, label in enumerate(self.args.labels_list)}

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        word_tokens = []
        for i in range(len(preds_list)):
            w_log = self._convert_tokens_to_word_logits(
                out_input_ids[i], out_label_ids[i], out_attention_mask[i], token_logits[i],
            )
            word_tokens.append(w_log)

        model_outputs = [[word_tokens[i][j] for j in range(len(preds_list[i]))] for i in range(len(preds_list))]

        extra_metrics = {}
        for metric, func in kwargs.items():
            extra_metrics[metric] = func(out_label_list, preds_list)

        result = {
            "eval_loss": eval_loss,
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1_score": f1_score(out_label_list, preds_list),
            **extra_metrics,
        }

        results.update(result)

        os.makedirs(eval_output_dir, exist_ok=True)
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            if args.classification_report:
                cls_report = classification_report(out_label_list, preds_list, digits=4)
                writer.write("{}\n".format(cls_report))
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))

        if self.args.wandb_project and wandb_log:
            wandb.init(project=args.wandb_project, config={**asdict(args)}, **args.wandb_kwargs)

            labels_list = sorted(self.args.labels_list)

            truth = [tag for out in out_label_list for tag in out]
            preds = [tag for pred_out in preds_list for tag in pred_out]
            outputs = [np.mean(logits, axis=0) for output in model_outputs for logits in output]

            # ROC
            wandb.log({"roc": wandb.plots.ROC(truth, outputs, labels_list)})

            # Precision Recall
            wandb.log({"pr": wandb.plots.precision_recall(truth, outputs, labels_list)})

            # Confusion Matrix
            wandb.sklearn.plot_confusion_matrix(
                truth, preds, labels=labels_list,
            )

        return results, model_outputs, preds_list

    def predict(self, to_predict, split_on_space=True, prior_prec=1, gp_batch_size=2, candidate_num=5):
        """
        Performs predictions on a list of text.

        Args:
            to_predict: A python list of text (str) to be sent to the model for prediction.
            split_on_space: If True, each sequence will be split by spaces for assigning labels.
                            If False, to_predict must be a a list of lists, with the inner list being a
                            list of strings consisting of the split sequences. The outer list is the list of sequences to
                            predict on.

        Returns:
            preds: A Python list of lists with dicts containing each word mapped to its NER tag.
            model_outputs: A Python list of lists with dicts containing each word mapped to its list with raw model output.
        """  # noqa: ignore flake8"

        device = self.device
        model = self.model
        model.is_pred = True
        args = self.args
        pad_token_label_id = self.pad_token_label_id
        preds = None

        if split_on_space:
            predict_examples = [
                InputExample(i, sentence.split(), [self.args.labels_list[0] for word in sentence.split()])
                for i, sentence in enumerate(to_predict)
            ]


        eval_dataset = self.load_and_cache_examples(None, to_predict=predict_examples)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset,
                                     sampler=eval_sampler,
                                     batch_size=args.eval_batch_size)


        self._move_model_to_device()
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        model.eval()

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)


        eval_bacth_count = 0
        for batch in tqdm(eval_dataloader, disable=args.silent, desc="Running Prediction"):
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                inputs = self._get_inputs_dict(batch)
                """
                得到model的输出，在这里得到两个返回值
                这里重新 fit 这个 input
                outputs 是原来的输出
                before_last_layer_logits 是倒数第二层的输出 sequence_output
                """
                outputs, before_last_layer_logits = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                if self.args.n_gpu > 1:
                    print("self.args.n_gpu", self.args.n_gpu)
                    tmp_eval_loss = tmp_eval_loss.mean()
                    before_last_layer_logits = before_last_layer_logits.mean()
                    print("tmp_eval_loss", tmp_eval_loss)

                eval_loss += tmp_eval_loss.item()

            nb_eval_steps += 1

            # 将多个eval_batch里最后一层之前的拼合在一起
            if eval_bacth_count == 0:
                before_last_layer_total = before_last_layer_logits
            else:
                before_last_layer_total = torch.cat((before_last_layer_total, before_last_layer_logits), 0)

            eval_bacth_count += 1

            if preds is None:  # 在最开始循环的时候
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
                out_input_ids = inputs["input_ids"].detach().cpu().numpy()
                out_attention_mask = inputs["attention_mask"].detach().cpu().numpy()

            else:  # 在之后的循环
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
                out_input_ids = np.append(out_input_ids, inputs["input_ids"].detach().cpu().numpy(), axis=0)
                out_attention_mask = np.append(
                    out_attention_mask, inputs["attention_mask"].detach().cpu().numpy(), axis=0,
                )

            eval_loss = eval_loss / nb_eval_steps

        token_logits = preds

        preds = np.argmax(preds, axis=2)


        label_map = {i: label for i, label in enumerate(self.args.labels_list)}

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        if split_on_space:
            preds = [
                [{word: preds_list[i][j]} for j, word in enumerate(sentence.split()[: len(preds_list[i])])]
                for i, sentence in enumerate(to_predict)
            ]

        else:
            preds = [
                [{word: preds_list[i][j]} for j, word in enumerate(sentence[: len(preds_list[i])])]
                for i, sentence in enumerate(to_predict)
            ]
        # print("preds", preds)
        word_tokens = []
        for n, sentence in enumerate(to_predict):
            w_log = self._convert_tokens_to_word_logits(
                out_input_ids[n], out_label_ids[n], out_attention_mask[n], token_logits[n],
            )
            word_tokens.append(w_log)



        if split_on_space:
            model_outputs = [
                [{word: word_tokens[i][j]} for j, word in enumerate(sentence.split()[: len(preds_list[i])])]
                for i, sentence in enumerate(to_predict)
            ]


        # 定义新的loader
        """
        GP的过程
        在这里进行了修改
        """
        token_id_dict = token_length_and_first_token_id(out_label_ids, out_input_ids)             # 获取每条 data 的每个 word 的 first_token 对应的位置
        before_last_layer_total = before_last_layer_total.clone().detach()                        # 获取最后一层线性层之前的数据，并使数据脱离计算图
        gp_loader = DataLoader(before_last_layer_total, batch_size=gp_batch_size, shuffle=True)   # 新建用于 GP 的 data_loader
        model_last_layer = model.classifier                                                       # 提取 BertForTokenClassification 的最后一层线性层

        post_prec = compute_laplace(model=model_last_layer,                        # 获取后验分布的 variance
                                    train_loader=gp_loader,                        # compute_laplace 来源于 DNN2GP
                                    prior_prec=prior_prec,
                                    device=self.device)

        variances = compute_dnn2gp_quantities(model=model_last_layer,              # 计算模型需要输出的 variance
                                  data_loader=gp_loader,                           # compute_dnn2gp_quantities 来源于 DNN2GP
                                  device=self.device,
                                  post_prec=post_prec)


        word_mean = []
        word_variance = []
        word_top_k_index = []
        for i in range(len(variances)):                                             # i 代表第 i 条 input_data
            variance = variances[i]                                                 # variance 是每条 data 对应的 variance  (128, 120)
            mean = token_logits[i, :, :]                                            # mean 是每条 data 对应的 mean          (128, 120)
            first_token_id_total = token_id_dict["first_token_id_total"][i]         # first tokens 的 id                  (words_number)
            first_token_variance = variance[first_token_id_total, :]                # first tokens 对应的 variance         (words_number, 120)
            first_token_mean = mean[first_token_id_total, :]                        # first tokens 对应的 mean             (words_number, 120)

            top_k_index = np.argsort(-first_token_mean, axis=1)[:,:var_num]         # 找到前 K 个有最大 mean 的 label id     (words_num, K)

            # 新建 word_mean_top_k, word_variance_top_k 两个新的矩阵，维度和 top_k_index 一样的
            # 将 top_k_index 对应的 mean 和 variance 储存到两个新的矩阵中
            word_mean_top_k = np.empty(top_k_index.shape)                           # 将这 K 个 label 对应的 mean 储存到新的矩阵里
            word_variance_top_k = np.empty(top_k_index.shape)                       # 将这 K 个 label 对应的 variance 储存到新的矩阵里
            for n in range(len(first_token_id_total)):
                for m in range(var_num):
                    word_mean_top_k[n, m] = first_token_mean[n, top_k_index[n, m]]           # 前 K 个最大 mean 的 label 对应的 mean
                    word_variance_top_k[n, m] = first_token_variance[n, top_k_index[n, m]]   # 前 K 个最大 mean 的 label 对应的 variance

            # 只保留前 K 个最大 mean 的 label 对应的 mean, variance 和 label_id
            word_top_k_index.append(top_k_index)                   # shape: (data_num [word_num, var_num])
            word_mean.append(word_mean_top_k)                      # shape: (data_num [word_num, var_num])
            word_variance.append(word_variance_top_k)              # shape: (data_num [word_num, var_num])

            word_outputs = {"word_top_k_index":word_top_k_index,       # 返回的是一个字典
                            "word_mean":word_mean,
                            "word_variance":word_variance}

        return preds, word_outputs




    def _convert_tokens_to_word_logits(self, input_ids, label_ids, attention_mask, logits):

        ignore_ids = [
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token),
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token),
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token),
        ]

        # Remove unuseful positions
        masked_ids = input_ids[(1 == attention_mask)]
        masked_labels = label_ids[(1 == attention_mask)]
        masked_logits = logits[(1 == attention_mask)]
        for id in ignore_ids:
            masked_labels = masked_labels[(id != masked_ids)]
            masked_logits = masked_logits[(id != masked_ids)]
            masked_ids = masked_ids[(id != masked_ids)]

        # Map to word logits
        word_logits = []
        tmp = []
        for n, lab in enumerate(masked_labels):
            if lab != self.pad_token_label_id:
                if n != 0:
                    word_logits.append(tmp)
                tmp = [list(masked_logits[n])]
            else:
                tmp.append(list(masked_logits[n]))
        word_logits.append(tmp)

        return word_logits

    def load_and_cache_examples(self, data, evaluate=False, no_cache=False, to_predict=None):
        """
        这部分相当于是读取数据的
        Reads data_file and generates a TensorDataset containing InputFeatures. Caches the InputFeatures.
        Utility function for train() and eval() methods. Not intended to be used directly.

        Args:
            data: Path to a .txt file containing training or evaluation data OR a pandas DataFrame containing 3 columns - sentence_id, words, labels.
                    If a DataFrame is given, each sentence should be split into words, with each word assigned a tag, and with all words from the same sentence given the same sentence_id.
            evaluate (optional): Indicates whether the examples are for evaluation or for training.
            no_cache (optional): Force feature conversion and prevent caching. I.e. Ignore cached features even if present.


        调用的部分
        train_dataset = self.load_and_cache_examples(train_data)
        eval_dataset = self.load_and_cache_examples(None, to_predict=predict_examples)

        """

        process_count = self.args.process_count

        tokenizer = self.tokenizer
        args = self.args

        if not no_cache:
            no_cache = args.no_cache

        mode = "dev" if evaluate else "train"  # examples 就是训练用的数据

        if to_predict:  # 如果有训练数据
            examples = to_predict  # A python list of text (str) to be sent to the model for prediction.
            no_cache = True  # 是否没有储存的地方

        cached_features_file = os.path.join(  # cache_features 保存的路径
            args.cache_dir,
            "cached_{}_{}_{}_{}_{}".format(
                mode, args.model_type, args.max_seq_length, self.num_labels, len(examples),
            ),
        )
        if not no_cache:  # 设定cache的储存地址
            os.makedirs(self.args.cache_dir, exist_ok=True)

        if os.path.exists(cached_features_file) and (
                (not args.reprocess_input_data and not no_cache)
                or (mode == "dev" and args.use_cached_eval_features and not no_cache)
        ):
            features = torch.load(cached_features_file)  # 导入torch保存的features数据
            logger.info(f" Features loaded from cache at {cached_features_file}")
        else:
            logger.info(" Converting to features started.")
            features = convert_examples_to_features(  # 提取从examples中保存的参数
                examples,  # features是一个list

                self.args.labels_list,
                self.args.max_seq_length,
                self.tokenizer,
                # XLNet has a CLS token at the end
                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                sep_token=tokenizer.sep_token,
                # RoBERTa uses an extra separator b/w pairs of sentences,
                # cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                sep_token_extra=args.model_type in MODELS_WITH_EXTRA_SEP_TOKEN,
                # PAD on the left for XLNet
                pad_on_left=bool(args.model_type in ["xlnet"]),
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                pad_token_label_id=self.pad_token_label_id,
                process_count=process_count,
                silent=args.silent,
                use_multiprocessing=args.use_multiprocessing,
                chunksize=args.multiprocessing_chunksize,
                mode=mode,
                use_multiprocessing_for_evaluation=args.use_multiprocessing_for_evaluation,
            )

            if not no_cache:  # 如果 no_cache，保存 feature 到 cache_features 保存的路径
                torch.save(features, cached_features_file)
        # features 里面有 input_ids, input_mask, segment_ids, label_ids
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

        if self.args.model_type == "layoutlm":
            all_bboxes = torch.tensor([f.bboxes for f in features], dtype=torch.long)

        if self.args.onnx:
            return all_label_ids

        if self.args.model_type == "layoutlm":
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_bboxes)
        else:
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        return dataset

    def convert_to_onnx(self, output_dir=None, set_onnx_arg=True):  # 转为onnx的形式
        """Convert the model to ONNX format and save to output_dir

        Args:
            output_dir (str, optional): If specified, ONNX model will be saved to output_dir (else args.output_dir will be used). Defaults to None.
            set_onnx_arg (bool, optional): Updates the model args to set onnx=True. Defaults to True.
        """  # noqa
        if not output_dir:
            output_dir = os.path.join(self.args.output_dir, "onnx")
        os.makedirs(output_dir, exist_ok=True)

        if os.listdir(output_dir):
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Output directory for onnx conversion must be empty.".format(output_dir)
            )

        onnx_model_name = os.path.join(output_dir, "onnx_model.onnx")

        with tempfile.TemporaryDirectory() as temp_dir:
            self.save_model(output_dir=temp_dir, model=self.model)

            convert(
                framework="pt",
                model=temp_dir,
                tokenizer=self.tokenizer,
                output=Path(onnx_model_name),
                pipeline_name="ner",
                opset=11,
            )

        self.args.onnx = True
        self.tokenizer.save_pretrained(output_dir)
        self.config.save_pretrained(output_dir)
        self._save_model_args(output_dir)

    def _move_model_to_device(self):
        self.model.to(self.device)

    def _get_last_metrics(self, metric_values):
        return {metric: values[-1] for metric, values in metric_values.items()}

    def _get_inputs_dict(self, batch):  # 返回 inputs 这个字典
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[3],
        }
        # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
        if self.args.model_type in ["bert", "xlnet", "albert", "layoutlm"]:
            inputs["token_type_ids"] = batch[2]

        if self.args.model_type == "layoutlm":
            inputs["bbox"] = batch[4]

        return inputs

    def _create_training_progress_scores(self, **kwargs):  # 评分
        extra_metrics = {key: [] for key in kwargs}
        training_progress_scores = {
            "global_step": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "train_loss": [],
            "eval_loss": [],
            **extra_metrics,
        }

        return training_progress_scores

    def save_model(self, output_dir=None, optimizer=None, scheduler=None, model=None, results=None):
        """
        模型会保存什么
        """
        if not output_dir:  # 定义模型保存的路径
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if model and not self.args.no_save:
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)  # 保存的是"pytorch_model.bin" 和 "config.json"
            self.tokenizer.save_pretrained(
                output_dir)  # 保存的是"special_tokens_map.json" "tokenizer_config.json" "vocab.txt"
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))  # "training_args.bin" 保存的是 self.args
            if optimizer and scheduler and self.args.save_optimizer_and_scheduler:
                torch.save(optimizer.state_dict(),
                           os.path.join(output_dir, "optimizer.pt"))  # "optimizer.pt" 保存的是 optimizer.state_dict(),
                torch.save(scheduler.state_dict(),
                           os.path.join(output_dir, "scheduler.pt"))  # "scheduler.pt" 保存的是 scheduler.state_dict()
            self._save_model_args(output_dir)  # "model_args.json" 保存模型参数

        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

    def _save_model_args(self, output_dir):  # 保存模型，写入 "model_args.json"
        os.makedirs(output_dir, exist_ok=True)
        self.args.save(output_dir)  # 详见 .config.ModelArgs.save(self, output_dir)

    def _load_model_args(self, input_dir):  # 加载模型参数
        args = NERArgs()
        args.load(input_dir)  # # 详见 .config.ModelArgs.load(self, output_dir)
        return args

    def get_named_parameters(self):
        return [n for n, p in self.model.named_parameters()]
