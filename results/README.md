### Training details for autoencoder <br>
Vocab_size = 20_000 <br>
Dataset subset = 600,000 sentences <br>
Batch size = 64 <br>
Epochs = 7 <br>
Maximum sentence length = 28 <br>
encoder_dim = 100 <br>
decoder_dim = 600 <br>
latent_dim = 100 <br>
dropout_prob = 0.5 <br>
ae_learning_rate = 1e-3 <br>
Optimizer = Adam <br>
Betas = (0.9, 0.999) <br>
word_embedding = 200 <br>
Lambda = 0.2 <br>
Loss = (1 - Lambda) * Reconstruction_error + Lambda * Encoder_loss (as given by Oshri and Khandwala) <br> 

### Training details for GAN <br>
Data seubset = 200,000 sentences <br>
Batch size = 64 <br>
Epochs = 30 <br>
Layer_dim = 100 <br>
n_layers = 10  <br>
critic_lr = generator_lr = 1e-4 <br>
Optimizer = Adam <br>
Betas = (0.5, 0.9) <br>
