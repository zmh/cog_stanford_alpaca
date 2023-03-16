from typing import List, Optional
from cog import BasePredictor, Input
import os

class Predictor(BasePredictor):

    def predict(
        self,
        model_path: str = Input(description="path to model"),
        tokenizer_path: str = Input(description="path to tokenizer"),
        data_path: str = Input(description="path to data", default='alpaca_data.json'),
        output_path: str = Input(description="path to save model", default='alpaca_out')
        ) -> int:
        if not output_path.startswith('/src'):
            output_path = os.path.join('src', output_path)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        command = f'''torchrun --nproc_per_node=4 --master_port=9292 train.py \
            --model_name_or_path {model_path} \
            --tokenizer_name_or_path {tokenizer_path} \
            --data_path {data_path} \
            --bf16 True \
            --output_dir {output_path} \
            --num_train_epochs 1 \
            --per_device_train_batch_size 4 \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps 1 \
            --evaluation_strategy "no" \
            --save_strategy "steps" \
            --save_steps 2000 \
            --learning_rate 2e-5 \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --fsdp "full_shard auto_wrap" \
            --fsdp_transformer_layer_cls_to_wrap 'LLaMADecoderLayer' \
            --tf32 True '''
        res = os.system(command)
        return res
        