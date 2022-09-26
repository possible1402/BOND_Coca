warmup_steps=200
learning_rate=0.00001
self_training_period=200
self_training_begin_step=600
model_type=roberta
self_training_reinit=0
self_training_label_mode=hard
self_training_hp_label_category=1
num_train_epochs=5
OUTPUT=outputs/conll03/self_training/${model_type}_reinit${self_training_reinit}_begin${self_training_begin_step}_period${self_training_period}_${self_training_label_mode}_hplabelcategory${self_training_hp_label_category}_${num_train_epochs}_${learning_rate}/
python ../run_self_training_ner.py \
--data_dir dataset/conll03_distant/ \
--model_type ${model_type} \
--model_name_or_path roberta-base \
--learning_rate ${learning_rate} \
--weight_decay 1e-4 \
--adam_epsilon 1e-8 \
--adam_beta1 0.9 \
--adam_beta2 0.98 \
--num_train_epochs ${num_train_epochs} \
--warmup_steps ${warmup_steps} \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 32 \
--logging_steps 100 \
--save_steps 100000 \
--do_train \
--do_eval \
--do_predict \
--evaluate_during_training \
--output_dir ${OUTPUT} \
--seed 42 \
--max_seq_length 128 \
--overwrite_output_dir \
--self_training_reinit ${self_training_reinit} \
--self_training_begin_step ${self_training_begin_step} \
--self_training_label_mode ${self_training_label_mode} \
--self_training_period ${self_training_period} \
--self_training_hp_label_category ${self_training_hp_label_category} \
--visible_device 0 \
--label_index 1 \
--whether_category_oriented \

