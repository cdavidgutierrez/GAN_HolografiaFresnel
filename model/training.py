import tensorflow as tf
import numpy as np
import  matplotlib.pyplot as plt

import time
import cv2

from model import *
from IPython import display

generator = model.Generator()
discriminator = model.Discriminator()

def generate_images(model, test_input, tar):
    """
    Function to plot some images during training.
    """
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(np.squeeze(display_list[i]) * 0.5 + 0.5)
        plt.axis('off')
        #print(np.unique(np.squeeze(display_list[i])))
    plt.show()

def to_test_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [tf.expand_dims(test_input[0], axis=0), tf.expand_dims(tar[0], axis=0), tf.expand_dims(prediction[0], axis=0)]
    """
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(np.squeeze(display_list[i]) * 0.5 + 0.5)
        plt.axis('off')
    plt.show()
    """

    return np.squeeze(display_list)

@tf.function
def train_step(input_image, target, step, summary_writer):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = model.generator_loss(disc_generated_output, gen_output, target)
        disc_loss = model.discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                discriminator.trainable_variables)

    model.generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    model.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
        tf.summary.scalar('disc_loss', disc_loss, step=step//1000)

def fit(train_ds, test_ds, steps, checkpoint, summary_writer, checkpoint_prefix):
    example_input, example_target = next(iter(test_ds.take(1)))
    start = time.time()

    #borrar.
    #score = []
    #pasos = []

    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
        if (step) % 1000 == 0:
            display.clear_output(wait=True)

            if step != 0:
                print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

            start = time.time()

            generate_images(generator, example_input, example_target)
            print(f"Step: {step//1000}k")

        train_step(input_image, target, step, summary_writer)

        # Training step
        if (step+1) % 10 == 0:
            print('.', end='', flush=True)


        # Save (checkpoint) the model every 5k steps
        if (step + 1) % 5000 == 0 or step==0:
            fake_target_images = [to_test_images(generator, inp, tar)[2] for inp, tar in train_ds.take(100)]
            real_target_images = [to_test_images(generator, inp, tar)[1] for inp, tar in train_ds.take(100)]
            zeros = np.zeros_like(fake_target_images[0])
            fake_target_images = [cv2.merge((zeros, zeros, img)) for img in fake_target_images]
            real_target_images = [cv2.merge((zeros, zeros, img)) for img in real_target_images]
            # geval = EvalGAN(real_target_images, fake_target_images)
            # fid = geval.calculate_fid()
            # score.append(fid)
            # pasos.append(step)

            checkpoint.save(file_prefix=checkpoint_prefix)
    #return pasos, score
