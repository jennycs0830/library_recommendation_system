import argparse
from clearml import Dataset

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_project", type=str, default="AI_recommender")
    parser.add_argument("--dataset_name", type=str, default="books_with_intro")
    parser.add_argument("--upload_files", nargs="+", type=str)

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    dataset = Dataset.create(dataset_project=args.dataset_project, dataset_name=args.dataset_name)
    
    for file in args.upload_files:
        dataset.add_files(file)
    
    dataset.upload()
    dataset.finalize()

if __name__ == "__main__":
    main()