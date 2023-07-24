#!/usr/bin/env bash

mbert_path=./mbert
src_lang=en

# xlmr_path=../trained-transformers/xlmr-base

for seed in 1111 2222 3333 4444 5555; do
    for tgt_lang in fr es nl ru; do
python main.py --tfm_type mbert \
            --exp_type smt \
            --model_name_or_path $mbert_path \
            --data_dir ./data \
            --src_lang $src_lang \
            --tgt_lang $tgt_lang \
            --do_train \
            --do_eval \
            --ignore_cached_data \
            --per_gpu_train_batch_size 16 \
            --per_gpu_eval_batch_size 12 \
            --learning_rate 5e-5 \
            --tagging_schema BIEOS \
            --overwrite_output_dir \
            --max_steps 2000 \
            --train_begin_saving_step 1500 \
            --eval_begin_end 1500-2000 \
            --seed $seed

python main.py --tfm_type mbert \
            --exp_type teacher_one \
            --model_name_or_path $mbert_path \
            --data_dir ./data \
            --src_lang $src_lang \
            --tgt_lang $tgt_lang \
            --do_train \
            --do_eval \
            --ignore_cached_data \
            --per_gpu_train_batch_size 16 \
            --per_gpu_eval_batch_size 12 \
            --learning_rate 5e-5 \
            --tagging_schema BIEOS \
            --overwrite_output_dir \
            --max_steps 2000 \
            --train_begin_saving_step 1500 \
            --eval_begin_end 1500-2000 \
            --seed $seed

python main.py --tfm_type mbert \
            --exp_type teacher_two \
            --model_name_or_path $mbert_path \
            --data_dir ./data \
            --src_lang $src_lang \
            --tgt_lang $tgt_lang \
            --do_train \
            --do_eval \
            --ignore_cached_data \
            --per_gpu_train_batch_size 16 \
            --per_gpu_eval_batch_size 12 \
            --learning_rate 5e-5 \
            --tagging_schema BIEOS \
            --overwrite_output_dir \
            --max_steps 2000 \
            --train_begin_saving_step 1500 \
            --eval_begin_end 1500-2000 \
            --seed $seed

python main.py --tfm_type mbert \
            --exp_type teacher_three \
            --model_name_or_path $mbert_path \
            --data_dir ./data \
            --src_lang $src_lang \
            --tgt_lang $tgt_lang \
            --do_train \
            --do_eval \
            --ignore_cached_data \
            --per_gpu_train_batch_size 16 \
            --per_gpu_eval_batch_size 12 \
            --learning_rate 5e-5 \
            --tagging_schema BIEOS \
            --overwrite_output_dir \
            --max_steps 2000 \
            --train_begin_saving_step 1500 \
            --eval_begin_end 1500-2000 \
            --seed $seed

python main_cl_cse_sentiment_level.py --tfm_type mbert \
            --exp_type smt \
            --model_name_or_path $mbert_path \
            --data_dir ./data \
            --src_lang $src_lang \
            --tgt_lang $tgt_lang \
            --do_train \
            --do_eval \
            --ignore_cached_data \
            --per_gpu_train_batch_size 64 \
            --per_gpu_eval_batch_size 12 \
            --learning_rate 5e-5 \
            --tagging_schema BIEOS \
            --overwrite_output_dir \
            --max_steps 2000 \
            --train_begin_saving_step 1500 \
            --eval_begin_end 1500-2000 \
            --seed $seed

python main_cl_cse_sentiment_level.py --tfm_type mbert \
            --exp_type teacher_one \
            --model_name_or_path $mbert_path \
            --data_dir ./data \
            --src_lang $src_lang \
            --tgt_lang $tgt_lang \
            --do_train \
            --do_eval \
            --ignore_cached_data \
            --per_gpu_train_batch_size 64 \
            --per_gpu_eval_batch_size 12 \
            --learning_rate 5e-5 \
            --tagging_schema BIEOS \
            --overwrite_output_dir \
            --max_steps 2000 \
            --train_begin_saving_step 1500 \
            --eval_begin_end 1500-2000 \
            --seed $seed

python main_cl_cse_sentiment_level.py --tfm_type mbert \
            --exp_type teacher_two \
            --model_name_or_path $mbert_path \
            --data_dir ./data \
            --src_lang $src_lang \
            --tgt_lang $tgt_lang \
            --do_train \
            --do_eval \
            --ignore_cached_data \
            --per_gpu_train_batch_size 64 \
            --per_gpu_eval_batch_size 12 \
            --learning_rate 5e-5 \
            --tagging_schema BIEOS \
            --overwrite_output_dir \
            --max_steps 2000 \
            --train_begin_saving_step 1500 \
            --eval_begin_end 1500-2000 \
            --seed $seed

python main_cl_cse_sentiment_level.py --tfm_type mbert \
            --exp_type teacher_three \
            --model_name_or_path $mbert_path \
            --data_dir ./data \
            --src_lang $src_lang \
            --tgt_lang $tgt_lang \
            --do_train \
            --do_eval \
            --ignore_cached_data \
            --per_gpu_train_batch_size 64 \
            --per_gpu_eval_batch_size 12 \
            --learning_rate 5e-5 \
            --tagging_schema BIEOS \
            --overwrite_output_dir \
            --max_steps 2000 \
            --train_begin_saving_step 1500 \
            --eval_begin_end 1500-2000 \
            --seed $seed

python main_cl_cse_token_level.py --tfm_type mbert \
            --exp_type smt \
            --model_name_or_path $mbert_path \
            --data_dir ./data \
            --src_lang $src_lang \
            --tgt_lang $tgt_lang \
            --do_train \
            --do_eval \
            --ignore_cached_data \
            --per_gpu_train_batch_size 32 \
            --per_gpu_eval_batch_size 12 \
            --learning_rate 5e-5 \
            --tagging_schema BIEOS \
            --overwrite_output_dir \
            --max_steps 2000 \
            --train_begin_saving_step 1500 \
            --eval_begin_end 1500-2000 \
            --seed $seed

python main_cl_cse_token_level.py --tfm_type mbert \
            --exp_type teacher_one \
            --model_name_or_path $mbert_path \
            --data_dir ./data \
            --src_lang $src_lang \
            --tgt_lang $tgt_lang \
            --do_train \
            --do_eval \
            --ignore_cached_data \
            --per_gpu_train_batch_size 32 \
            --per_gpu_eval_batch_size 12 \
            --learning_rate 5e-5 \
            --tagging_schema BIEOS \
            --overwrite_output_dir \
            --max_steps 2000 \
            --train_begin_saving_step 1500 \
            --eval_begin_end 1500-2000 \
            --seed $seed

python main_cl_cse_token_level.py --tfm_type mbert \
            --exp_type teacher_two \
            --model_name_or_path $mbert_path \
            --data_dir ./data \
            --src_lang $src_lang \
            --tgt_lang $tgt_lang \
            --do_train \
            --do_eval \
            --ignore_cached_data \
            --per_gpu_train_batch_size 32 \
            --per_gpu_eval_batch_size 12 \
            --learning_rate 5e-5 \
            --tagging_schema BIEOS \
            --overwrite_output_dir \
            --max_steps 2000 \
            --train_begin_saving_step 1500 \
            --eval_begin_end 1500-2000 \
            --seed $seed

python main_cl_cse_token_level.py --tfm_type mbert \
            --exp_type teacher_three \
            --model_name_or_path $mbert_path \
            --data_dir ./data \
            --src_lang $src_lang \
            --tgt_lang $tgt_lang \
            --do_train \
            --do_eval \
            --ignore_cached_data \
            --per_gpu_train_batch_size 32 \
            --per_gpu_eval_batch_size 12 \
            --learning_rate 5e-5 \
            --tagging_schema BIEOS \
            --overwrite_output_dir \
            --max_steps 2000 \
            --train_begin_saving_step 1500 \
            --eval_begin_end 1500-2000 \
            --seed $seed
    done
done
