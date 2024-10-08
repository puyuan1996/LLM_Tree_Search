set -e
export TEST_NO_TERMINAL=1
# export TEST_WITH_TERMINAL=1
# export TEST_COT_GREEDY=1
# export TEST_COT_SC=1

# export CUDA_VISIBLE_DEVICES=4,5,6,7


# CT2_DIR={your ct2 model cache}
# CRITIC_PATH={your critic model cache}


CT2_DIR=/mnt/afs/niuyazhe/code/LLM_Tree_Search/llama-2-7b-hf-sft-ct2/llama2_sft_ep3_ct2
CRITIC_PATH=/mnt/afs/niuyazhe/data/llama-2-7b-hf

# torchrun --nproc_per_node=1 --master-port 29503 ../../tsllm/offline_rl/test_sft_and_v.py \
#     --ct2_dir $CT2_DIR \
#     --critic_model_path $CRITIC_PATH \
#     --tokenizer_path $CRITIC_PATH \
#     --save_dir $1/pi_sftep3_v_sftep1 \
#     --env_name gsm8k \
#     --test True


# 原始的
torchrun --nproc_per_node=1 --master-port 29503 -m pdb ../../tsllm/offline_rl/test_sft_and_v.py \
    --ct2_dir $CT2_DIR \
    --critic_model_path $CRITIC_PATH \
    --tokenizer_path $CRITIC_PATH \
    --save_dir $1/pi_sftep3_v_sftep1 \
    --env_name gsm8k \
    --test True
