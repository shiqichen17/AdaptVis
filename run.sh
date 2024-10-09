# export CUDA_VISIBLE_DEVICES='3,4'
# export TEST_MODE=True
# python3 main_aro.py --dataset=Controlled_Images_A --model-name='llava1.5' --download --method=scaling_vis --eval out --weight=0.8   --option=four
python3 main_aro.py --dataset=Controlled_Images_A --model-name='llava1.5' --download --method adapt_vis --weight1 0.5  --weight2 1.5 --threshold 0.4 --option=four