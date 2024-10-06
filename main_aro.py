import argparse
import os
import pandas as pd
import pdb
from torch.utils.data import DataLoader

from model_zoo import get_model
from dataset_zoo import get_dataset
from misc import seed_all, _default_collate, save_scores

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--model-name", default="openai-clip:ViT-B/32", type=str, \
            choices=["openai-clip:ViT-B/32", "openai-clip:ViT-L/14", \
                "NegCLIP", "laion-clip:roberta-ViT-B/32", \
                "coca", "xvlm-pretrained-4m", "xvlm-pretrained-16m", \
                "blip-base-14m", "blip-base-129m", "flava", \
                "coca-cap", "xvlm-flickr", "xvlm-coco", \
                "blip-flickr-base", "blip-coco-base", "llava1.5","llava1.6","llama"])
    parser.add_argument("--dataset", default="VG_Relation", type=str, \
            choices=["VG_Relation", "VG_Attribution", "COCO_Order", \
            "Flickr30k_Order", "Controlled_Images_A", "Controlled_Images_B", \
            "COCO_QA_one_obj", "COCO_QA_two_obj", "VG_QA_one_obj", "VG_QA_two_obj"])
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--mode",  type=str)
    
    parser.add_argument("--download", action="store_true", help="Whether to download the dataset if it doesn't exist. (Default: False)")
    parser.add_argument("--save-scores", action="store_true", help="Whether to save the scores for the retrieval to analyze later.")
    parser.add_argument("--output-dir", default="./outputs", type=str)
    return parser.parse_args()


def main(args):

    
    
    seed_all(args.seed)
    
    model, image_preprocess = get_model(args.model_name, args.device)
    
    dataset = get_dataset(args.dataset, image_preprocess=image_preprocess, download=args.download)
    # pdb.set_trace()
    # For some models we just pass the PIL images, so we'll need to handle them in the collate_fn. 
    collate_fn = _default_collate if image_preprocess is None else None
    
    joint_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    scores = model.get_retrieval_scores_batched(joint_loader,args.mode)
    result_records = dataset.evaluate_scores(scores,args.output_dir,args.dataset,args.mode)
    
    for record in result_records:
        record.update({"Model": args.model_name, "Dataset": args.dataset, "Seed": args.seed})
    
    output_file = os.path.join(args.output_dir, f"{args.dataset}_{args.model_name.replace('/', '_')}.csv")
    df = pd.DataFrame(result_records)
    print(f"Saving results to {output_file}")
    if os.path.exists(output_file):
        all_df = pd.read_csv(output_file, index_col=0)
        all_df = pd.concat([all_df, df])
        all_df.to_csv(output_file)

    else:
        df.to_csv(output_file)
        
    if args.save_scores:
        save_scores(scores, args)

    
if __name__ == "__main__":
    args = config()
    main(args)
