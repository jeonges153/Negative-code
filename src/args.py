import argparse

def parse_args():
    
    parser = argparse.ArgumentParser()
    
    # DIRECTORY
    parser.add_argument(
        '--from_csv', '-fc', type=str, default="./only_3_345711.csv",
        help='Directory where the csv file.')
    parser.add_argument(
        '--dir_data', '-dd', type=str, default="./only_3_345711_text.json",
        help='Directory where the dataset will be stored.')
    parser.add_argument(
        '--inference_data', '-id', type=str, default='./345711_inference.csv',
        help='Directory where the inference dataset will be stored.'
    )
    parser.add_argument(
        '--dir_out', '-do', type=str, default='out',
        help='Directory where the output will be stored.')
    parser.add_argument(
        '--dir_checkpoint', '-dc', type=str, default='./checkpoint/',
        help='Directory where the checkpoint stored.')
    parser.add_argument(
        '--dir_wandb', '-dw', type=str, default='wandb',
        help='Directory what the wandb name is.')
    
    # TRAINING
    parser.add_argument(
        '--epoch', '-e', type=int, default=50,
        help='Number of training steps.')
    parser.add_argument(
        '--batch_size', '-b', type=int, default=8,
        help='Batch size for training.')
    # parser.add_argument(
    #     '--make_batch', '-bs', type=int, default=1,
    #     help='Batch size for training.')
    parser.add_argument(
        '--num_class', '-n', type=int, default=5,
        help='Number of class.')
    parser.add_argument(
        '--category_list', '-cl', type=list, default=[3, 4, 5, 7, 11],
        help='Category list for datasets')
    parser.add_argument(
        '--temperature', '--t', type=float, default=0.1,
        help='Temperature for training.')
    parser.add_argument(
        '--alpha', '--a', type=float, default=0.7,
        help='Weight for contrastive learning.')
    parser.add_argument(
        '--lr_base', '-lrb', type=float, default=1e-3)
    parser.add_argument(
        '--seed', '-se', type=int, default=42,
        help='Seed for reproducibility.')
    parser.add_argument(
        '--shuffle', '-sh', type=bool, default=False,
        help='Dataset shuffle or not.')
    
    
    args = parser.parse_args()
    
    return args