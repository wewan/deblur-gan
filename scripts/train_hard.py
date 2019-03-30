import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import datetime
import click
import numpy as np
import tqdm
import sys
# sys.path.insert(0,'/var/scratch/wwang/deblur/keras/deblur-gan')
from deblurgan.utils import load_images, write_log
from deblurgan.losses import wasserstein_loss, perceptual_loss
from deblurgan.model import generator_model, discriminator_model, generator_containing_discriminator_multiple_outputs

from keras.callbacks import TensorBoard
from keras.optimizers import Adam

BASE_DIR = 'weights/'

#####################

def save_all_weights(d, g, epoch_number, current_loss):
    now = datetime.datetime.now()
    save_dir = os.path.join(BASE_DIR, '{}{}'.format(now.month, now.day))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    g.save_weights(os.path.join(save_dir, 'generator_{}_{}.h5'.format(epoch_number, current_loss)), True)
    d.save_weights(os.path.join(save_dir, 'discriminator_{}.h5'.format(epoch_number)), True)


def train_multiple_outputs(n_images, batch_size, log_dir, epoch_num, critic_updates=5):
    data = load_images('../images/train', n_images)
    y_train, x_train = data['B'], data['A']

    g = generator_model()
    d = discriminator_model()
    d_on_g = generator_containing_discriminator_multiple_outputs(g, d)

    d_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    d_on_g_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    d.trainable = True
    d.compile(optimizer=d_opt, loss=wasserstein_loss)
    d.trainable = False
    loss = [perceptual_loss, wasserstein_loss]
    loss_weights = [100, 1]
    d_on_g.compile(optimizer=d_on_g_opt, loss=loss, loss_weights=loss_weights)
    d.trainable = True


    log_path = './logs'
    tensorboard_callback = TensorBoard(log_path)

    hn_num = int(batch_size*0.5)
    hp_num = int(batch_size*0.5)

    output_true_batch, output_false_batch = np.ones((batch_size, 1)), -np.ones((batch_size, 1))
    hard_true_batch, hard_false_batch = np.ones((batch_size+hp_num, 1)), -np.ones((batch_size+hn_num, 1))


    for epoch in tqdm.tqdm(range(epoch_num)):
        permutated_indexes = np.random.permutation(x_train.shape[0])

        d_losses = []
        d_on_g_losses = []
        for index in range(int(x_train.shape[0] / batch_size)):
            batch_indexes = permutated_indexes[index*batch_size:(index+1)*batch_size]
            image_blur_batch = x_train[batch_indexes]
            image_full_batch = y_train[batch_indexes]

            generated_images = g.predict(x=image_blur_batch, batch_size=batch_size)


            ##############
            # d.trainable = False
            # temp_hn = []
            # for i in range(generated_images.shape[0]):
            #     t_s = d.predict(generated_images[i])
            #     temp_hn.append(t_s)
            # hn_ind = np.argsort(temp_hn)[::-1][:hn_num]
            #
            # hard_neg  = generated_images[hn_ind]
            # hard_neg_y= image_full_batch[hn_ind]
            # hard_pos = []
            # hard_pos = []
            #
            # neg_train= np.concatenate((generated_images,hard_neg),axis=0)
            # pos_train= np.concatenate((image_full_batch,hard_neg_y),axis=0)

            for _ in range(critic_updates):
                d.trainable = False
                temp_hn = []
                for i in range(generated_images.shape[0]):
                    t_s = d.predict(generated_images[i][np.newaxis,...])[0][0]
                    temp_hn.append(t_s)
                hn_ind = np.argsort(temp_hn)[::-1][:hn_num]

                hard_neg = generated_images[hn_ind]
                hard_neg_y = image_full_batch[hn_ind]
                hard_pos = []
                hard_pos = []
                d.trainable = True
                neg_train = np.concatenate((generated_images, hard_neg), axis=0)
                pos_train = np.concatenate((image_full_batch, hard_neg_y), axis=0)
                d_loss_real = d.train_on_batch(pos_train, hard_true_batch)
                d_loss_fake = d.train_on_batch(neg_train, hard_false_batch)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
                d_losses.append(d_loss)

            # for _ in range(critic_updates):
            #     d_loss_real = d.train_on_batch(image_full_batch, output_true_batch)
            #     d_loss_fake = d.train_on_batch(generated_images, output_false_batch)
            #     d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
            #     d_losses.append(d_loss)

            # #################################
            # d.trainable = False
            # for i in range(generated_images.shape[0]):
            #     t_s = d.predict(generated_images[i][np.newaxis, ...])[0][0]
            #     temp_hn.append(t_s)
            # hn_ind = np.argsort(temp_hn)[:hn_num]
            #
            # hard_g_x = image_blur_batch[hn_ind]
            # hard_g_y = image_full_batch[hn_ind]
            # g_blur = np.concatenate((image_blur_batch, hard_g_x), axis=0)
            # g_full = np.concatenate((image_full_batch, hard_g_y), axis=0)
            # d_on_g_loss = d_on_g.train_on_batch(g_blur, [g_full, hard_true_batch])
            # d_on_g_losses.append(d_on_g_loss)
            #
            # d.trainable = True
                ##############################
            d.trainable = False

            d_on_g_loss = d_on_g.train_on_batch(image_blur_batch, [image_full_batch, output_true_batch])
            d_on_g_losses.append(d_on_g_loss)

            d.trainable = True

        # write_log(tensorboard_callback, ['g_loss', 'd_on_g_loss'], [np.mean(d_losses), np.mean(d_on_g_losses)], epoch_num)
        print(np.mean(d_losses), np.mean(d_on_g_losses))
        with open('log.txt', 'a+') as f:
            f.write('{} - {} - {}\n'.format(epoch, np.mean(d_losses), np.mean(d_on_g_losses)))

        save_all_weights(d, g, epoch, int(np.mean(d_on_g_losses)))


@click.command()
# @click.option('--n_images', default=-1, help='Number of images to load for training')
@click.option('--n_images', default=10, help='Number of images to load for training')
@click.option('--batch_size', default=4, help='Size of batch')
# @click.option('--log_dir', required=True, help='Path to the log_dir for Tensorboard')
@click.option('--log_dir', default='./log', help='Path to the log_dir for Tensorboard')
@click.option('--epoch_num', default=4, help='Number of epochs for training')
@click.option('--critic_updates', default=5, help='Number of discriminator training')
def train_command(n_images, batch_size, log_dir, epoch_num, critic_updates):
    return train_multiple_outputs(n_images, batch_size, log_dir, epoch_num, critic_updates)


if __name__ == '__main__':
    train_command()
