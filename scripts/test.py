import sys
sys.path.insert(0,'/var/scratch/wwang/deblur/keras/deblur-gan')

import numpy as np
from PIL import Image
import click
import tensorflow as tf
from deblurgan.model import generator_model
from deblurgan.utils import load_images, deprocess_image
from deblurgan.metrics import SSIM,PSNR
import tqdm



def test(batch_size):
    data = load_images('../images/test', 300)
    y_test, x_test = data['B'], data['A']
    g = generator_model()
    # g.load_weights('./weights/331/generator_3_1538.h5')
    # g.load_weights('./weights_hard/331/generator_3_1746.h5')
    # g.load_weights('../generator3-22.h5')
    g.load_weights('../deblur-20.h5')

    # im1 = tf.decode_png('../images/test/A/GOPR0384_11_00_000001.png')
    # im2 = tf.decode_png('../images/test/B/GOPR0384_11_00_000001.png')
    # ssim = tf.image.ssim(im1, im2, max_val=255)
    # print(ssim)
    psnr =0
    ssim =0
    for index in tqdm.tqdm(range(int(300 / batch_size))):
        batch_test = x_test[index * batch_size:(index + 1) * batch_size]
        batch_label = y_test[index * batch_size:(index + 1) * batch_size]


        generated_images = g.predict(x=batch_test, batch_size=batch_size)
        generated = np.array([deprocess_image(img) for img in generated_images])
        batch_test = deprocess_image(batch_test)
        batch_label = deprocess_image(batch_label)

        for i in range(generated_images.shape[0]):
            y = batch_label[i, :, :, :]
            x = batch_test[i, :, :, :]
            img = generated[i, :, :, :]
            # with tf.Session() as sess:
            #     sess.run(tf.initialize_all_variables())
            #     yy = tf.convert_to_tensor(y, dtype=tf.float32)
            #     imgimg = tf.convert_to_tensor(img, dtype=tf.float32)
            #     ssim = tf.image.ssim(yy, imgimg, max_val=255)
            #     psnr = tf.image.psnr(yy,imgimg,max_val=255)
            #     sess.run(psnr)
            # with tf.Session() as sess:
            #     sess.run(tf.initialize_all_variables())  # execute init_op
            #     # print the random values that we sample
            #     psnr += sess.run(psnr)
            #     ssim += sess.run(ssim)

            # print(ssim)
            # print(psnr)
            # yy= np.transpose(y[np.newaxis,...],(0,3,1,2))
            # imgimg = np.transpose(img[np.newaxis,...],(0,3,1,2))
            psnr += PSNR(y,img)
            # print(psnr)
            # ssim += SSIM(yy,imgimg)

            output = np.concatenate((y, x, img), axis=1)
            im = Image.fromarray(output.astype(np.uint8))
            im.save('results{}.png'.format(i))

    print(psnr/300)
    # print(ssim/generated_images.shape[0])


@click.command()
@click.option('--batch_size', default=4, help='Number of images to process')
def test_command(batch_size):
    return test(batch_size)


if __name__ == "__main__":
    test_command()
