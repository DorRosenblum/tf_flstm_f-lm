"""
Entry point for training and eval
"""
import os

import tensorflow as tf
from data_utils import Vocabulary, Dataset
from language_model import LM
from run_utils import run_train, run_eval

tf.flags.DEFINE_string("logdir", "lm1b", "Logging directory.")
tf.flags.DEFINE_string("datadir", "1-billion-word-language-modeling-benchmark-master", "DataSet directory.")
tf.flags.DEFINE_string("mode", "train", "Whether to run 'train' or 'eval' model.")
tf.flags.DEFINE_string("hpconfig", "", "Overrides default hyper-parameters.")
tf.flags.DEFINE_integer("num_gpus", 0, "Number of GPUs used.")
tf.flags.DEFINE_integer("eval_steps", 70, "Number of eval steps.")

FLAGS = tf.flags.FLAGS


def main(_):
    """
    Start either train or eval. Note hardcoded parts of path for training and eval data
    """
    hps = LM.get_default_hparams().parse(FLAGS.hpconfig)
    hps._set("num_gpus", FLAGS.num_gpus)
    print ('*****HYPER PARAMETERS*****')
    print (hps)
    print ('**************************')

    print('\x1b[6;30;42m' + '~~~~~>>Almog&Dor debug: our training DataSetDir=%s  , LogDir=%s' % (FLAGS.datadir,FLAGS.logdir) + '\x1b[0m')

    vocab = Vocabulary.from_file(os.path.join(FLAGS.datadir, "1b_word_vocab.txt"))

    if FLAGS.mode == "train":
        #hps.batch_size = 256
        dataset = Dataset(vocab, os.path.join(FLAGS.datadir,
                                              "training-monolingual.tokenized.shuffled/*"))

        trainlogdir=(FLAGS.logdir+str("/")+"train")#(FLAGS.logdir+str("\\")+"train")#os.path.join(FLAGS.logdir, "train")
        print('\x1b[6;30;42m' + '~~~~~>>Almog&Dor debug: train log dir=%s' % (trainlogdir) + '\x1b[0m')

        run_train(dataset, hps, trainlogdir, ps_device="/gpu:0")
        print('\x1b[6;30;41m' + '~~>>Almog&Dor debug: Finished run_train !!!!!!!!!!!' + '\x1b[0m')
    elif FLAGS.mode.startswith("eval_"):
        print('\x1b[6;30;42m' + '~~~~~>>Almog&Dor debug: eval mode' + '\x1b[0m')


        if FLAGS.mode.startswith("eval_train"):
            data_dir = os.path.join(FLAGS.datadir, "training-monolingual.tokenized.shuffled/*")
        elif FLAGS.mode.startswith("eval_full"):
            data_dir = os.path.join(FLAGS.datadir, "heldout-monolingual.tokenized.shuffled/*")
        else:
            data_dir = os.path.join(FLAGS.datadir, "heldout-monolingual.tokenized.shuffled/news.en.heldout-00000-of-00050")
        dataset = Dataset(vocab, data_dir, deterministic=True)
        run_eval(dataset, hps, FLAGS.logdir, FLAGS.mode, FLAGS.eval_steps)

if __name__ == "__main__":
    tf.app.run()
