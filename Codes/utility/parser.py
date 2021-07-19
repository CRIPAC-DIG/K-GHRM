import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', nargs='?', default='../Data/',
                        help='Input data path.')

    parser.add_argument('--word_topk', type=int, default=40,
                        help='word_topk')
    parser.add_argument('--ent_topk', type=int, default=9,
                        help='ent_topk')
    parser.add_argument('--fold', type=int, default=1,
                        help='run fold 1~5')

    parser.add_argument('--dataset', nargs='?', default='robust04',
                        help='Choose a dataset from {robust04, clueweb09}')

    parser.add_argument('--epoch', type=int, default=300,
                        help='Number of epoch.')
    parser.add_argument('--doc_len', type=int, default=300,
                        help='DOC_PAD_LEN')
    parser.add_argument('--qrl_len', type=int, default=4,
                        help='QRL_PAD_LEN')
    parser.add_argument('--ent_num', type=int, default=50,
                        help='ENT_PAD_LEN')
    parser.add_argument('--drop_rate', type=float, default=0.2,
                        help='DROP_RATE')                           
    parser.add_argument('--idf', type=int, default=1,
                        help='idf_flag')
    parser.add_argument('--layers', type=int, default=2,
                        help='num of layers')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size.')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--dp', type=float, default=0,
                        help='dropout rate.')
    parser.add_argument('--l2', type=float, default=0,
                        help='l2 norm.')
    parser.add_argument('--model', nargs='?', default='KGIR',
                        help='Specify the name of model.')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='')
    parser.add_argument('--report', type=int, default=1,
                        help='')
    return parser.parse_args()
