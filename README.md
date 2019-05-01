# VAE
Variational Autoencoder for generating faces similar to frey face dataset.  
Autoencoders are neural nets that learn to encode image into f.ex. 10-dimensional vector and then decode from this vector back to original image.  
Variational Autoencoders guarantee probability distribution of this 10-D space, which allows for generating new images.  
Here is a nice [lecture](https://www.youtube.com/watch?v=uaaqyVS9-rM) explaining it  
  
This VAE learns on 2000 face images of Brendan Frey and generates new ones.  
Code was based on [this](https://github.com/pytorch/examples/tree/master/vae) pytorch example

### Some generated faces:
<img src="resultsResults/sample100_3D.png" width="450"/>

### Original / encoded-decoded comparison
<img src="resultsResults/encodeComparison100_3D.png" width="450"/>


