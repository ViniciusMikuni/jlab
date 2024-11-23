from tensorflow import keras
import tensorflow as tf

from keras import layers
import tensorflow.keras.backend as K

def get_discriminator(num_inputs):
    discriminator = keras.Sequential(
        [
        keras.Input(shape=(num_inputs)),
        layers.Dense(32,activation=layers.LeakyReLU(0.2)),
        layers.Dense(1, activation="sigmoid"),
        ],
        name="discriminator",
    )
    return discriminator



def get_generator(num_inputs):
    generator = keras.Sequential(
        [
        keras.Input(shape=(num_inputs)),
        layers.Dense(32,activation=layers.LeakyReLU(0.2)),
        layers.Dense(num_inputs,activation="sigmoid"),
        ],
        name="generator",
    )
    return generator


class GAN(keras.Model):
    def __init__(self, discriminator, generator,num_features=1):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator       
        self.num_features = num_features

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name="discriminator loss")
        self.g_loss_metric = keras.metrics.Mean(name="generator loss")

    @property
    def metrics(self):
        #Add metrics to be printed at the end of each epoch
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, data,alpha=0.01):
        # Sample random points in the latent space
        batch_size = tf.shape(data)[0]
        #add a bit of noise to the data inputs
        noise = tf.random.uniform(
            shape=(batch_size, self.num_features)
        )
        data += alpha*noise
        
        random_inputs = tf.random.normal(
            shape=(batch_size, self.num_features)
        )

        # Call the generator
        generated_data = self.generator(random_inputs)

        # Combine them with real images
        combined_data = tf.concat([generated_data, data], axis=0)

        # Create labels for the classifier
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # Add random noise to the labels to improve stability
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_data)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        random_inputs = tf.random.normal(
            shape=(batch_size, self.num_features)
        )

        fake_labels = tf.zeros((batch_size, 1))

        # Train the generator 
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_inputs))
            g_loss = self.loss_fn(fake_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "discriminator loss": self.d_loss_metric.result(),
            "generator loss": self.g_loss_metric.result(),
        }

    @tf.function
    def generate(self,nevts):        
        random_inputs = tf.random.normal(
                shape=(nevts, self.num_features)
            )
        return self.generator(random_inputs)
