import argparse

import nltk

from coatt.coattention_experiment_runner import CoattentionNetExperimentRunner


if __name__ == "__main__":
    # Feel free to add more args, or change/remove these.
    parser = argparse.ArgumentParser(description='Load VQA.')
    parser.add_argument('--model', type=str, choices=['simple', 'coattention'], default='simple')
    parser.add_argument('--pretrained_embed', type=str, default='word2vec_vi_words_300dims')

    parser.add_argument('--train_image_dir', type=str, default='./datasets/vigqa/train')
    parser.add_argument('--train_question_path', type=str,
                        default='./datasets/vigqa/vqa/vigqa_train_questions.json')
    parser.add_argument('--train_annotation_path', type=str,
                        default='./datasets/vigqa/vqa/vigqa_train_annotations.json')

    parser.add_argument('--val_image_dir', type=str, default='./datasets/vigqa/val')
    parser.add_argument('--val_question_path', type=str,
                        default='./datasets/vigqa/vqa/vigqa_val_questions.json')
    parser.add_argument('--val_annotation_path', type=str,
                        default='./datasets/vigqa/vqa/vigqa_val_annotations.json')

    parser.add_argument('--test_image_dir', type=str, default='./datasets/vigqa/test')
    parser.add_argument('--test_question_path', type=str,
                        default='./datasets/vigqa/vqa/vigqa_test_questions.json')
    parser.add_argument('--test_annotation_path', type=str,
                        default='./datasets/vigqa/vqa/vigqa_test_annotations.json')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--num_data_loader_workers', type=int, default=0)
    args = parser.parse_args()

    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')

    if args.model == "simple":
        experiment_runner_class = SimpleBaselineExperimentRunner
    elif args.model == "coattention":
        experiment_runner_class = CoattentionNetExperimentRunner
    else:
        raise ModuleNotFoundError()

    experiment_runner = experiment_runner_class(train_image_dir=args.train_image_dir,
                                                train_question_path=args.train_question_path,
                                                train_annotation_path=args.train_annotation_path,
                                                val_image_dir=args.val_image_dir,
                                                val_question_path=args.val_question_path,
                                                val_annotation_path=args.val_annotation_path,
                                                test_image_dir=args.test_image_dir,
                                                test_question_path=args.test_question_path,
                                                test_annotation_path=args.test_annotation_path,
                                                batch_size=args.batch_size,
                                                num_epochs=args.num_epochs,
                                                num_data_loader_workers=args.num_data_loader_workers,
                                                pretrained_embed=args.pretrained_embed,
                                                lr=args.lr)
    experiment_runner.train()
    experiment_runner.test()