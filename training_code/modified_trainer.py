#Ohter python programs
import models
import evaluation

#Needed libraries
import numpy as np
import tensorflow as tf
from absl import logging

#Handles the OS
import os

#Might be used if we want to test with symbolic
#from enum import Enum

#This is how they do it might move it down
tfkl = tf.keras.layers

class training_temp:
    def cosine_decay(self, step, warmup_steps=1000):
        warmup_factor = min(step, warmup_steps) / warmup_steps
        decay_step = max(step - warmup_steps, 0) / (
        self.number_training_iterations - warmup_steps)
        return self.learning_rate * warmup_factor * (1 + tf.cos(decay_step * np.pi)) / 2
    
    
    def eval_step(self, vision_model, model_head, dataset_list,
              dataset_measurement_tags):
        """Evaluate distribution-based metrics.

        Args:
          vision_model: The model which produces a feature vector to hand to IPDF.
          model_head: The IPDF model.
          dataset_list: A list of datasets, to evaluate separately.
          dataset_measurement_tags: The names to associate with each dataset in
            dataset_list, when outputting metrics.
        Returns:
          A dictionary of values indexed by their associated descriptor tag.
        """
        assert len(dataset_list) == len(dataset_measurement_tags)
        measurements = {}
        logging.info('Started eval_step.')
        for dataset, tag in zip(dataset_list, dataset_measurement_tags):
            avg_log_likelihood, spread = evaluation.eval_spread_and_loglikelihood(
                vision_model,
                model_head,
                dataset,
                batch_size=self.batch_size,
                skip_spread_evaluation=self.skip_spread_evaluation,
                number_eval_iterations=self.number_eval_iterations)
            measurements[f'gt_log_likelihood_{tag}'] = avg_log_likelihood
            measurements[f'spread_{tag}']        = spread
        return measurements
    
    
    @tf.function
    def train_step(self, vision_model, model_head, optimizer, images, rotations_gt):
        with tf.GradientTape() as tape:
            vision_description = vision_model(images, training=True)
            loss = model_head.compute_loss(vision_description, rotations_gt)
        grads = tape.gradient(
            loss,
            vision_model.trainable_variables + model_head.trainable_variables)
        optimizer.apply_gradients(
        zip(grads, vision_model.trainable_variables +
            model_head.trainable_variables))
        return loss
    
        
    def __init__(self):
        self.output_dir = "/home/steffen/uni/sem2mas/proj/training/output_folder"
        self.checkpoint_dir  = "/home/steffen/uni/sem2mas/proj/training/checkpoint"
        self.number_training_iterations = 100 #10000
        self.learning_rate = 1e-4
        
        self.batch_size = 32
        self_batch_size_test = 32
        
        self.number_eval_iterations = None  #('number_eval_iterations', None,
                                            #'The number of iterations to eval.')
        
        self.head_network_specs = [256]*2 #input image size
        self.downsample_continuous_gt = 0
        self.number_eval = None
        self.optimizer = 'Adam'
        self.number_train_queries = 2**12
        self.number_eval_queries = 2**16
        self.so3_sampling_mode = ['random', 'grid'] #Grid is avalible
        self.number_fourier_components = 0 # controls the positions Ignroed for now
        
        
        self.model_head = None
        
        self.MUCK = False   #flags.DEFINE_bool('mock', False,
                            #'Skip download of dataset and pre-trained weights. '
                            #'Useful for testing.')
        self.skip_spread_evaluation = False #('skip_spread_evaluation', False, 'Whether to skip the '
                                            #'evaluation of the spread metric, which can be slow for '
                                            #'shapes with many ground truths.')
        self.eval_every = -1    # (eval_every', -1, 'How often to evaluate.  
                                #   If -1, evaluate 100 times during training.)
                            
        self.save_model = True  #'Whether to save the vision and IPDF'
                                #' models at the end of training.')
        
        self.tf_optimizer = None
        self.tf_learn = None 
        
        
        self.data_set            = []
        self.data_set_training   = []
        self.data_set_validation = []
        
        
        self.dset_val_list = [] #List is the values
        self.dset_val_tags = [] #Tags would be [CUBE]
    def start_stuff(self):
        #Get the pretrained Resnet model.
        self.vision_model, self.len_visual_description = models.create_vision_model()
        
#        model_head = models.ImplicitSO3(len_visual_description,
#                                FLAGS.number_fourier_components,
#                                FLAGS.head_network_specs,
#                                FLAGS.so3_sampling_mode,
#                                FLAGS.number_train_queries,
#                                FLAGS.number_eval_queries)
        self.model_head = models.ImplicitSO3(self.len_visual_description,
                                    self.number_fourier_components,
                                    self.head_network_specs,
                                    self.so3_sampling_mode,
                                    self.number_train_queries,
                                    self.number_eval_queries)
        
        #This is for the symsol it might be added after
        #dset_train = data.load_symsol('[grid]', mode='train', mock=self.MUCK)
        #dset_train = dset_train.repeat().shuffle(1000).batch(self.batch_size)
        
        #Data parser is needed here.
        visualization_images, visualization_rotations_gt = [[], []]
        for image, rotations_gt in tf.data.experimental.sample_from_datasets(
                self.dset_val_list).take(8):
            visualization_images.append(image)
            visualization_rotations_gt.append(rotations_gt)

        measurement_labels = []
        for tag in self.dset_val_tags:
            measurement_labels += [f'gt_log_likelihood_{tag}', f'spread_{tag}']
        measurements = {}
        
        
        
        
        
        self.tf_optimizer = tf.keras.optimizers.get(self.optimizer)
        self.tf_optimizer.learning_rate = self.learning_rate
        train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        
        
        
        log_dir = os.path.join(self.output_dir, 'logs')
        train_summary_writer = tf.summary.create_file_writer(log_dir)
                
        checkpoint = tf.train.Checkpoint(
        vision_model=self.vision_model, model_head=self.model_head, optimizer=self.tf_optimizer)

        checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

        max_ckpts_to_keep = 1
        checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, directory=self.checkpoint_dir, max_to_keep=max_ckpts_to_keep)
        
        logging.info('Started training.')
        for batch_data in self.dset_train:
            step_num = self.tf_optimizer.iterations.numpy()
            if step_num > self.number_training_iterations:
                break
            
            tf.keras.backend.set_value(self.tf_optimizer.learning_rate,
                               self.cosine_decay(step_num))
            images, rotations_gt = batch_data
            loss = self.train_step(self.vision_model, self.model_head, self.tf_optimizer, images, rotations_gt)

            train_loss(loss)
            
            
            if (step_num) % 100 == 0:
                avg_loss = train_loss.result()
                train_loss.reset_states()
                with train_summary_writer.as_default():
                  tf.summary.scalar('loss', avg_loss, step=step_num)
                  tf.summary.scalar(
                      'learning_rate', self.tf_optimizer.learning_rate, step=step_num)
            logging.info('Step %d, training loss=%.2f', step_num, avg_loss)

            if (step_num+1) % self.eval_every == 0:
                measurements = self.eval_step(
                    self.vision_model,
                    self.model_head,
                    self.dset_val_list,
                    self.dset_val_tags)
                
                with train_summary_writer.as_default():
                    logline = f'Step {step_num}: '
                    for k, v in measurements.items():
                      tf.summary.scalar(k, v, step=step_num)
                      logline += f'{k}={v:.2f} '
                    logging.info(logline)
                    logging.info('Started visualize_so3.')
                    distribution_images = evaluation.visualize_model_output(
                        self.vision_model, self.model_head, visualization_images,
                        visualization_rotations_gt)
                    tf.summary.image('output_distribution', distribution_images,
                                     step=step_num)
  
        if self.save_model:
            visual_input = tfkl.Input(shape=(self.len_visual_description,))
            query_input = tfkl.Input(shape=(None, self.model_head.len_query,))
            inp = [visual_input, query_input]
            saveable_head_model = tf.keras.Model(inp, self.model_head(inp))
            self.save_model = False
        if not self.save_model:
            self.vision_model.save(os.path.join(self.output_dir, 'base_vision_model'))
            saveable_head_model.save(os.path.join(self.output_dir, 'ipdf_head_model'))

            logging.info('Saved models.')
            
            
        

test_if_works= training_temp()

test_if_works.start_stuff()