from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import os
import skimage.io as skio
import sys
import tensorflow as tf

def create_sprite_image(images):
    SPRITE_MAX_WIDTH, SPRITE_MAX_HEIGHT = 8192, 8192
    n, h, w, c = images.shape
    """
    # Constraint: n_cols * w = n_rows * h
    #             n_cols * n_rows = n
    n_rows = np.sqrt((w / float(h)) * n)
    n_cols = n / n_rows
    assert n_rows == int(n_rows) and n_cols == int(n_cols)
    n_rows, n_cols = int(n_rows), int(n_cols)
    sprite_image = np.zeros((n_rows * h, n_cols * w, c))
    """
    n_cols = SPRITE_MAX_WIDTH // w
    n_rows = int(np.ceil(n / n_cols))
    assert n_rows * h <= SPRITE_MAX_HEIGHT
    sprite_image = np.zeros((n_cols * w, n_cols * w, c), dtype=np.uint8)
    for row in range(n_rows):
        for col in range(n_cols):
            if (row * n_cols + col) >= n:
                break
            sprite_image[row*h:(row + 1)*h, col*w:(col+1)*w] = \
                images[row * n_cols + col]
    if c == 1:
        sprite_image = np.dstack([sprite_image] * 3)
    return sprite_image

path_to_embeddings = sys.argv[1]
LOG_DIR = sys.argv[2]

sess = tf.Session()

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

embeddings_npy = np.load(path_to_embeddings)

embedding_var = tf.Variable(embeddings_npy, name='embedding')
config = projector.ProjectorConfig()

embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name

if len(sys.argv) >= 4 and sys.argv != 'None':
    # write labels to metadata file
    labels = np.load(sys.argv[3])
    embedding.metadata_path = 'metadata.tsv'
    with open(LOG_DIR + '/' + embedding.metadata_path, 'w') as f:
        for y in labels:
            f.write(str(int(y)) + '\n')

if len(sys.argv) >= 5:
    # create sprite image
    # REQUIRED: Choose number of images and width/height such that a square
    #           grid can be made out of them.
    # Format: (N, H, W, C)
    images = np.load(sys.argv[4])
    sprite_image = create_sprite_image(images)
    embedding.sprite.image_path = 'sprite.png'
    print(sprite_image.shape)
    embedding.sprite.single_image_dim.extend([images.shape[2], images.shape[1]])
    skio.imsave(LOG_DIR + '/' + embedding.sprite.image_path, sprite_image)
        
summary_writer = tf.summary.FileWriter(LOG_DIR)

projector.visualize_embeddings(summary_writer, config)

init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()
saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
