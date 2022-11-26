import numpy as np 
import os 
import cv2

from architecture import GenerativeAdversarialNetwork

MAIN_DIR = "tumor_images"

# Instantiate the Model
gan = GenerativeAdversarialNetwork()

generator = gan.build_generator()
discriminator = gan.build_discriminator()
gan_model = gan.generative_adversarial_network(discriminator=discriminator, generator=generator)

# Loading the Image data
def load_images(folder):
    
    imgs = []
    for i in os.listdir(folder):
        img_dir = os.path.join(folder,i)
        try:
            img = cv2.imread(img_dir)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Grayscale
            img = cv2.resize(img, (128,128)) # Resize to 128x128
            imgs.append(img)
        except:
            continue
        
    imgs = np.array(imgs)
    return imgs

data = load_images(MAIN_DIR)

# get 20 random numbers to get 20 random images
idxs = np.random.randint(0, 155, 20)
X_train = data[idxs]

# Normalize the Images
X_train = (X_train.astype(np.float32) - 127.5) / 127.5

# Reshape images 
X_train = X_train.reshape(-1, gan.WIDTH, gan.HEIGHT, gan.CHANNELS)

# Training Process
np.random.seed(gan.SEED)
for epoch in range(gan.EPOCHS):
    for batch in range(gan.STEPS_PER_EPOCH):

        # Generating the noise and creating fake images
        noise = np.random.normal(0,1, size=(gan.BATCH_SIZE, gan.NOISE_DIM))
        fake_X = generator.predict(noise)
        
        # Get real images from the data
        idx = np.random.randint(0, X_train.shape[0], size=gan.BATCH_SIZE)
        real_X = X_train[idx]

        # Put them together
        X = np.concatenate((real_X, fake_X))

        # Creating the labels
        disc_y = np.zeros(2*gan.BATCH_SIZE)
        disc_y[:gan.BATCH_SIZE] = 1

        # One step of gradient update on discriminator
        d_loss = discriminator.train_on_batch(X, disc_y)
        
        # Create all labels as real to fool the discriminator
        # One step of gradient update on the entire gan_model
        # Note: No changes on discriminator here, only Generator weights update 
        y_gen = np.ones(gan.BATCH_SIZE)
        g_loss = gan_model.train_on_batch(noise, y_gen)

    # Check Progress at every Epoch
    print(f"EPOCH: {epoch + 1} Generator Loss: {g_loss:.4f} Discriminator Loss: {d_loss:.4f}")