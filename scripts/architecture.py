from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, Reshape, Input, Conv2DTranspose
from keras.layers import LeakyReLU, Dropout
from keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')


class GenerativeAdversarialNetwork:

    def __init__(self):

        self.NOISE_DIM = 100  
        self.BATCH_SIZE = 4 
        self.STEPS_PER_EPOCH = 3750
        self.EPOCHS = 10
        self.SEED = 40
        self.WIDTH, self.HEIGHT, self.CHANNELS = 128, 128, 1
        self.OPTIMIZER = Adam(0.0002, 0.5)


    def build_generator(self):

        """
            Generator model "generates" images using random noise. The random noise AKA Latent Vector
            is sampled from a Normal Distribution which is given as the input to the Generator. Using
            Transposed Convolution, the latent vector is transformed to produce an image
            We use 3 Conv2DTranspose layers (which help in producing an image using features; opposite
            of Convolutional Learning)

            Input: Random Noise / Latent Vector
            Output: Image
        """

        model = Sequential([

            Dense(32*32*256, input_dim=self.NOISE_DIM),
            LeakyReLU(alpha=0.2),
            Reshape((32,32,256)),
            
            Conv2DTranspose(128, (4, 4), strides=2, padding='same'),
            LeakyReLU(alpha=0.2),

            Conv2DTranspose(128, (4, 4), strides=2, padding='same'),
            LeakyReLU(alpha=0.2),

            Conv2D(self.CHANNELS, (4, 4), padding='same', activation='tanh')
        ], 
        name="generator")
        model.compile(loss="binary_crossentropy", optimizer=self.OPTIMIZER)

        return model

    def build_discriminator(self):
        
        """
            Discriminator is the model which is responsible for classifying the generated images
            as fake or real. Our end goal is to create a Generator so powerful that the Discriminator
            is unable to classify real and fake images
            A simple Convolutional Neural Network with 2 Conv2D layers connected to a Dense output layer
            Output layer activation is Sigmoid since this is a Binary Classifier

            Input: Generated / Real Image
            Output: Validity of Image (Fake or Real)

        """

        model = Sequential([

            Conv2D(64, (3, 3), padding='same', input_shape=(self.WIDTH, self.HEIGHT, self.CHANNELS)),
            LeakyReLU(alpha=0.2),

            Conv2D(128, (3, 3), strides=2, padding='same'),
            LeakyReLU(alpha=0.2),

            Conv2D(128, (3, 3), strides=2, padding='same'),
            LeakyReLU(alpha=0.2),
            
            Conv2D(256, (3, 3), strides=2, padding='same'),
            LeakyReLU(alpha=0.2),
            
            Flatten(),
            Dropout(0.4),
            Dense(1, activation="sigmoid", input_shape=(self.WIDTH, self.HEIGHT, self.CHANNELS))
        ], name="discriminator")
        model.compile(loss="binary_crossentropy",
                            optimizer=self.OPTIMIZER)

        return model

    def generative_adversarial_network(self, discriminator, generator):

        """
        This is where we combine the generator and discriminator to create the Adversarial model.
        discriminator.trainable is set as False, since we don't want the discriminator to train at the same time as the generator.
        The Model class of Tensorflow helps put together the two neural networks.
        
        Input: RANDOM NOISE => shape: (100,) => Generator
        Output: Class of the Image: Fake(0) or Real(1) => Discriminator

        NOISE => Generator => IMAGE => Discriminator => OUTPUT CLASS
        """

        discriminator.trainable = False 

        gan_input = Input(shape=(self.NOISE_DIM,))
        fake_image = generator(gan_input)

        gan_output = discriminator(fake_image)

        gan = Model(gan_input, gan_output, name="gan_model")
        gan.compile(loss="binary_crossentropy", optimizer=self.OPTIMIZER)

        return gan
