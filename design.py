from absl import app
from absl import flags
from train import TrainPipeline
import os
from pre import initialize_weights, CC_index

flags.DEFINE_string('receptor_seq', None, 'SEQ of receptor')
flags.DEFINE_string('receptor_if_residues', None, 'Path to numpy array with receptor interface residues.')
flags.DEFINE_integer('peptide_length', None, 'Length of peptide binder.')
flags.DEFINE_string('output_dir', None, 'Path to a directory that will store the results.')
flags.DEFINE_bool('plDDT_only', None, 'Use plDDT as the only loss.')
flags.DEFINE_integer('num_iterations', None, 'Number of iterations to run.')
flags.DEFINE_string('receptor_name', None, 'Name of receptor.')
flags.DEFINE_integer('jumpout_num', None, 'Number of jumpout.')
FLAGS = flags.FLAGS

def main(argv):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    weights, peptide_sequence = initialize_weights(FLAGS.peptide_length)
    C2, C1 = CC_index(peptide_sequence)
    print("init seq", peptide_sequence)
    print("C1", C1, " C2", C2)
    training_pipeline = TrainPipeline(init_seq=peptide_sequence, receptor_seq=FLAGS.receptor_seq, pocket=FLAGS.receptor_if_residues, output_dir=FLAGS.output_dir, num_iterations=FLAGS.num_iterations, plDDT_only=FLAGS.plDDT_only, receptor_name=FLAGS.receptor_name, jumpout_num=FLAGS.jumpout_num)
    training_pipeline.run()

if __name__ == '__main__':
    app.run(main)