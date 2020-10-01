#-*-coding=utf-8-*-

from multiprocessing import Pool
import re
import tensorflow as tf
import common


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_shards', 10, "number of sharded tfrecord files.")
flags.DEFINE_string("term_index_file", "term_index.txt", "path of vocab file.")
flags.DEFINE_string("cid_name_file", "data/cid3_name.txt", "path of cid name and index file.")
flags.DEFINE_string("input_examples_file", "sample.all.dat", "path of traning data.")

if __name__ == '__main__':
    _, term_index = common.get_term_index(FLAGS.term_index_file)
    output_tfrecord_file_prefix = FLAGS.input_examples_file[:-4] + '.tfrecord'

    def _sample_to_tfrecord(shard_index):
        total_num_shards = FLAGS.num_shards
        output_tfrecord_file = output_tfrecord_file_prefix + '.' + "{:0>4d}_of_{:0>4d}".format(shard_index, total_num_shards)

        writer = tf.python_io.TFRecordWriter(output_tfrecord_file)
        num = 0
        with open(FLAGS.input_examples_file) as f:
            for i, line in enumerate(f):
                if i%total_num_shards != shard_index:
                    continue
                num += 1
                example = tf.train.SequenceExample()
                fl_terms = example.feature_lists.feature_list["terms"]
                fl_pos_cid = example.feature_lists.feature_list["pos_cid"]
                fl_neg_cid = example.feature_lists.feature_list["neg_cid"]
                fl_pos_pdt = example.feature_lists.feature_list["pos_pdt"]
                fl_neg_pdt = example.feature_lists.feature_list["neg_pdt"]
                tokens = line.strip().split('\t')
                if len(tokens) == 6:
                    terms = tokens[1].strip().split(',')[:8]
                    pos_cid = tokens[2].strip().split(',')[:6]
                    neg_cid = tokens[3].strip().split(',')[:6]
                    pos_brd = tokens[4].strip().split(',')[:5]
                    neg_brd = tokens[5].strip().split(',')[:5]
                    while len(terms) < 8:
                        terms.append('<PAD>')
                    while len(pos_cid) < 6:
                        pos_cid.append('<PAD>')
                    while len(neg_cid) < 6:
                        neg_cid.append('<PAD>')
                    while len(pos_pdt) < 2:
                        pos_pdt.append('<PAD>')
                    while len(neg_pdt) < 2:
                        neg_pdt.append('<PAD>')
                    for term in terms:
                        if term in term_index:
                            fl_terms.feature.add().int64_list.value.append(term_index[term])
                        else:
                            fl_terms.feature.add().int64_list.value.append(1)
                    for term in pos_cid:
                        if term in term_index:
                            fl_pos_cid.feature.add().int64_list.value.append(term_index[term])
                        else:
                            fl_pos_cid.feature.add().int64_list.value.append(1)
                    for term in neg_cid:
                        if term in term_index:
                            fl_neg_cid.feature.add().int64_list.value.append(term_index[term])
                        else:
                            fl_neg_cid.feature.add().int64_list.value.append(1)
                    for term in pos_pdt:
                        if term in term_index:
                            fl_pos_pdt.feature.add().int64_list.value.append(term_index[term])
                        else:
                            fl_pos_pdt.feature.add().int64_list.value.append(1)
                    for term in neg_pdt:
                        if term in term_index:
                            fl_neg_pdt.feature.add().int64_list.value.append(term_index[term])
                        else:
                            fl_neg_pdt.feature.add().int64_list.value.append(1)
                    writer.write(example.SerializeToString())
                if i % 1000000 == 0:
                    print('{i} line sample transfer succeed....'.format(i=i))
                    writer.flush()
        return num

    pool=Pool()
    results=pool.map(_sample_to_tfrecord, range(FLAGS.num_shards))
    print(results)

    print('train/dev examples txt to tfrecord transfering succeed...')
