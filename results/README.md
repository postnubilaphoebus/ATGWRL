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

Autoencoder loss graph: 
![AE_loss](https://github.com/postnubilaphoebus/ATGWRL/blob/main/results/Autoencoder%20loss%20after%2064000%20batches.png)

GAN loss graph: 
![GAN_loss](https://github.com/postnubilaphoebus/ATGWRL/blob/main/results/Plotted%20GAN%20loss%20after%20895batches%20(in%20100s).png)

Critic scores:
![Critic_scores](https://github.com/postnubilaphoebus/ATGWRL/blob/main/results/Plotted%20accs%20after%20895batches%20(in%20100s).png)

Example Outputs: <br> <br>

random seed: 0 <br> <br>

the that had had that it was . <br>
i was done and with it . <br>
he sent to him back . <br>
he never begged his back . <br>
there was all about about things she 'd about him . <br>
he was just the man . <br>
he would let to let it to to him . <br>
he could thought it was about it . <br>
she 'd have like the way way ? . <br>
luis laughed and and and opened him . <br>

random seed: 42 <br> <br>

he knew luis he n't believe it . <br>
luis had to bother that . . <br>
yes , luis was with the door and the time . <br>
she shook him him that of the the face . <br>
she was good really good . <br>
for he 'd thought for to to him ? <br>
he was myself and open before . <br>
he could n't give the man to . <br>
jase did n't think that . . <br>
he guided luis off jase forward . <br>



