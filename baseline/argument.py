
def parse_arguments_t(parser):
    # Training Hyperparameters
    parser.add_argument('--device', type=str, default="cuda", choices=['cpu', 'cuda'], help="GPU/CPU devices")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--digit2zero', action="store_true", default=False, help="convert the number to 0, make it true is better")
    parser.add_argument('--dataset', type=str, default="data")
    parser.add_argument('--optimizer', type=str, default="sgd")
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=1e-8)
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=32, help="default batch size is 32 (works well)")
    parser.add_argument('--num_epochs', type=int, default=30, help="Usually we set to 10.")  # origin 100
    parser.add_argument('--train_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--dev_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--num_outer_iterations', type=int, default=10, help="Number of outer iterations for cross validation")

    # bert hyperparameter
    parser.add_argument('--bert_model_dir', default='bert-base-chinese-pytorch', help="Directory containing the BERT model in PyTorch")
    parser.add_argument('--max_len', default=180, help="max allowed sequence length")
    parser.add_argument('--full_finetuning', default=True, action='store_false', help="Whether to fine tune the pre-trained model")
    parser.add_argument('--clip_grad', default=5, help="gradient clipping")

    # model hyperparameter
    parser.add_argument('--model_folder', type=str, default="saved_model", help="The name to save the model files")
    parser.add_argument('--log_name', type=str, default="train.log", help="The name to save the model files")

    args = parser.parse_args()
    return args
