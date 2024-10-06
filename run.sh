export CUDA_VISIBLE_DEVICES='3'
export TEST_MODE=True
datasets="Controlled_Images_A,Controlled_Images_B,COCO_QA_one_obj,COCO_QA_two_obj,VG_QA_one_obj,VG_QA_two_obj"


model_name='llava1.5'

eval='out'
methods='scaling_vis'
option=six
weights='0.5'

for dataset in $(echo $datasets | tr ',' ' ')
do
   
    python main_aro.py --dataset=$dataset --model-name=$model_name --download --mode $mode --method $method --eval $eval\
    --weight=$weight   --option=$option
    
done