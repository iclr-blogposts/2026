---
layout: distill
title: "Neural audio codecs: how to get audio into LLMs"
description: A look at why audio is harder to model than text and how we can make it easier with neural audio codecs. With a codec, we can turn audio into larger discrete tokens, train models to predict continuations for these tokens, and then decode those back into audio.
date: 2026-04-27
future: true
htmlwidgets: true
hidden: true

# Mermaid diagrams
mermaid:
  enabled: true
  zoomable: true

# Anonymize when submitting
authors:
  - name: Anonymous

# authors:
#   - name: Albert Einstein
#     url: "https://en.wikipedia.org/wiki/Albert_Einstein"
#     affiliations:
#       name: IAS, Princeton
#   - name: Boris Podolsky
#     url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
#     affiliations:
#       name: IAS, Princeton
#   - name: Nathan Rosen
#     url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
#     affiliations:
#       name: IAS, Princeton

# must be the exact same name as your blogpost
bibliography: 2026-04-27-neural-audio-codecs.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Text is easy
  - name: Sample by sample
  - name: Autoencoders with vector quantization (VQ-VAE)
  - name: Residual vector quantization
  - name: Now let’s tokenize audio
  - name: Why care about audio
  - name: Dealing with multiple levels
  - name: Finally, let’s train
  - name: How far can a codec get us?
  - name: Semantic tokens
  - name: Making poetry semantic
  - name: Semantic–acoustic tradeoff
  - name: Conclusion

_styles: >
  .audio-sample {
    width: 100%;
  }
  .bg-black {
    background-color: black;
  }
---

{% include video.liquid path="assets/img/2026-04-27-neural-audio-codecs/codec-intro.mp4" class="img-fluid rounded z-depth-1" controls=true muted=true autoplay=true loop=true %}
<div class="caption">
    The plan: sandwich a language model in an audio encoder/decoder pair (=neural
  audio codec), allowing it to predict audio continuations.
</div>

As of October 2025, speech LLMs suck. Many LLMs have voice interfaces, but they usually work by transcribing your speech, generating the answer in text, and using text-to-speech to read the response out loud. That’s perfectly fine in many cases, but it’s a wrapper, not _real_ speech understanding. The model can’t hear the frustration in your voice and respond with empathy, it can’t emphasize important words in its answer, it cannot sense sarcasm, and so on.

Yes, there _are_ LLMs
(Gemini 2.5 <d-cite key="DBLP:journals/corr/abs-2507-06261" /> <d-cite key="gemini_2_5_native_audio" />,
ChatGPT’s Advanced Voice Mode <d-cite key="DBLP:journals/corr/abs-2410-21276" />,
Qwen <d-cite key="DBLP:journals/corr/abs-2509-17765" />,
Moshi <d-cite key="DBLP:journals/corr/abs-2410-00037" />)
that understand and generate speech natively. But in practice, they’re either not as smart, or they behave like text model wrappers. Try asking any of them “Am I speaking in a low voice or a high voice?” in a high-pitched voice, and they won’t be able to tell you.

Clearly, speech LLMs lag behind text LLMs. But why? For text, we found out a few years ago that if you take a lot of text data, a big Transformer, and a lot of GPUs, you’ll get some pretty damn good text continuation models. Why can’t we just replace text with audio and get pretty damn good speech continuation models?

As a teaser, here’s what happens when you try to do that naively (warning, loud):

{% include audio.liquid path="assets/img/2026-04-27-neural-audio-codecs/nightmare-fuel-20lufs.wav" controls=true class="audio-sample" %}

We'll see how to make much better audio models using neural audio codecs, the de-facto standard way of getting audio into and out of LLMs.
With a codec, we can turn audio into larger discrete _tokens_, train models to predict continuations for these tokens, and then decode those back into audio: see animation above.

We’ll start from the basics and build up all the way to Mimi, a modern neural audio codec originally developed for Moshi <d-cite key="DBLP:journals/corr/abs-2410-00037" /> and later adopted by others for their models, notably Sesame’s CSM <d-cite key="sesame_uncanny_valley_voice" />.

## Text is easy

To tokenize text, everybody uses a technique called byte-pair encoding and rarely changes the tokenizer: OpenAI has been using [the same tokenizer](https://github.com/openai/tiktoken/blob/2ab6d3706d557b560b200be48e6a32324682c9a3/tiktoken/model.py#L8-L16C17) since GPT-4o <d-cite key="DBLP:journals/corr/abs-2410-21276" />, an ancient model if you count in LLM years.

{% include figure.liquid path="assets/img/2026-04-27-neural-audio-codecs/image.png" class="img-fluid rounded z-depth-1" %}
<div class="caption">
A random text from Wikipedia tokenized via the GPT-4o tokenizer
</div>

You can even get decent results without tokenizing text at all, just predicting individual
characters. One of the first posts that got me excited about machine learning was
Andrej Karpathy’s RNN effectiveness blog post <d-cite key="unreasonable_rnns" /> from 2015.
Karpathy trains a three-layer LSTM on a single GPU and gets
it to generate decent-looking code and LaTeX:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/2026-04-27-neural-audio-codecs/rnns-code.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/2026-04-27-neural-audio-codecs/rnns-latex.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Remember this was ten years ago, back when we didn’t even know that attention is all we need <d-cite key="DBLP:journals/corr/VaswaniSPUJGKP17" />.
Now compare Karpathy’s results to a sample from WaveNet <d-cite key="DBLP:conf/ssw/OordDZSVGKSK16" />, a model DeepMind published a year later:

{% include audio.liquid path="assets/img/2026-04-27-neural-audio-codecs/speaker-1.wav" controls=true class="audio-sample" %}

Purely acoustically, the audio sounds good, but it rarely even manages to produce a single correct English word. We can’t be too hard on WaveNet, though. The samples from Karpathy’s RNNs are only a few thousand characters long, but this 10-second audio consists of 160k audio samples, and WaveNet creates it by painstakingly predicting sample-by-sample.

{% include video.liquid path="assets/img/2026-04-27-neural-audio-codecs/wavenet-audio.mp4" class="img-fluid rounded z-depth-1"  controls=true muted=true autoplay=true loop=true %}
<div class="caption">
  A single second of audio consists of tens of thousands of samples, although it
  corresponds to just a few words. Animation from the WaveNet blog post <d-cite key="wavenet_blog" />
</div>

It’s difficult to build models that are coherent over time scales this long, and the model also takes very long to run for so many steps.

So instead of running the model to predict the samples one-by-one directly, we’d like to train a model to compress the audio into a more manageable size. We could compress the audio, use an LLM to predict a continuation in the compressed representation, and then decompress the result.

## Sample by sample

But first, let’s get a baseline model by generating audio sample by sample, like WaveNet does. The code for all of these experiments is open-source! [Link to code omitted for anonymization, reading the code is not necessary for understanding the blog post.] I forked Andrej Karpathy’s [nanoGPT](https://github.com/karpathy/nanoGPT) repo, a simple implementation of GPT-2.

Text and audio are kind of the same from the perspective of the language model: it’s just tokens in, tokens out. The only thing we need to do is to quantize the continuous values of the samples into discrete buckets. Like WaveNet, we’ll use the "μ-law algorithm" <d-cite key="ITU-G711" /> to get 256 buckets. We’ll treat those as 256 possible tokens.

Let’s train a language model on audio tokenized like this. For the dataset, we’ll use the Libri-Light dataset <d-cite key="DBLP:conf/icassp/KahnRZKXMKLCFLS20" />, following AudioLM <d-cite key="DBLP:journals/taslp/BorsosMVKPSRTGTZ23" />. Its train split contains 50k hours in total, but we’ll go with 1000 hours for this experiment. With this sample-by-sample tokenization, we end up with a dataset of 53 GB.

We train a small-ish transformer of 151.28M parameters, about the size of the smallest GPT-2 variant <d-cite key="radford2019language" />. When we sample from the model, it makes babbling sounds (warning, loud at times!):

{% include audio.liquid path="assets/img/2026-04-27-neural-audio-codecs/20250925_140600_3.wav" controls=true class="audio-sample" %}

Often, it goes into a “crackling mode” that it can’t seem to get out of:

{% include audio.liquid path="assets/img/2026-04-27-neural-audio-codecs/20250925_140600_4.wav" controls=true class="audio-sample" %}

I also trained a smaller model, which I teased at the beginning. It’s prone to generate nightmare fuel screeches (loud!):

{% include audio.liquid path="assets/img/2026-04-27-neural-audio-codecs/nightmare-fuel-2-20lufs.wav" controls=true class="audio-sample" %}

As you can tell, we’re not AGI yet. It sounds speech-like, but you can’t make out a single word and the voice keeps changing. No wonder: the context size of the model is 2048, which, for 16 kHz audio, translates to 128ms, not even the length of one word. Also, these 10-second examples took 30 minutes to generate on an H100, so we’re a few orders of magnitude away from being real-time.

So let’s build a neural audio codec to compress the audio. The hope is that if we reduce the sampling rate 100x, the model will also become “100x more coherent”. An old idea in machine learning is to do this using an _autoencoder:_ a model that takes an input, compresses it into a smaller “latent space”, and then tries to reconstruct the original input.

In our case, we’ll want an autoencoder whose latent space is quantized so that we can feed the latents into a language model and produce continuations. (You _can_ generate continuations with unquantized latents, but it’s trickier <d-cite key="DBLP:journals/corr/abs-2508-19205" /> <d-cite key="DBLP:journals/corr/abs-2509-06926" />.)

## Autoencoders with vector quantization (VQ-VAE)

Bear with me, because we’ll take a detour from audio: let’s build a quantized autoencoder on images from Fashion-MNIST <d-cite key="DBLP:journals/corr/abs-1708-07747" />. We’ll take a subset with the first three classes: t-shirt/top, trouser, and pullover.

{% include figure.liquid path="assets/img/2026-04-27-neural-audio-codecs/fashion-mnist-3.png" class="img-fluid rounded z-depth-1"%}
<div class="caption">
  image source: <d-cite key="DBLP:journals/information/NikfamCMM023" />
</div>

First, let’s train a regular autoencoder to encode the images into two-dimensional space:

{% include video.liquid path="assets/img/2026-04-27-neural-audio-codecs/vq_images_unquantized_v2.mp4" class="img-fluid rounded z-depth-1"  controls=true muted=true autoplay=true loop=true %}
<div class="caption">
  Training a regular autoencoder on Fashion-MNIST
</div>

Each frame shows one batch of training, with some batches skipped. The little images are the autoencoder’s reconstructions for the images in the batch. I’ve added colors for the three classes (t-shirt/top=blue trousers=yellow, pullover=purple), but the autoencoder doesn’t get a class as input – the space just naturally clusters by class. Let's zoom in on a few reconstructions:

{% include figure.liquid path="assets/img/2026-04-27-neural-audio-codecs/rvq-without-quantization-v4.png" class="img-fluid rounded z-depth-1 bg-black" %}
<div class="caption">
  Original images (top) and their reconstructed versions (bottom)
</div>

As you can tell, the reconstruction quality is not great. The images are blurry and the first two images are reconstructed to nearly the same thing. But we used a tiny network (4 fully connected layers for the encoder and decoder each) and projected into a mere two dimensions, so we can’t expect too much of our model.

Now let’s quantize these embeddings using a clustering. We’ll do something like k-means: we’ll maintain a list of the positions of the cluster centers. We initialize the positions randomly. For each training batch, we look at which embeddings would go to each cluster. (We don’t modify the embeddings, we just look at the assignment). Then we’ll nudge each cluster center towards the average position of these embeddings.

Also, if a center is unused for a while, we teleport it to a random embedding from the batch, because otherwise it has no way to get unstuck from its current position.

{% include video.liquid path="assets/img/2026-04-27-neural-audio-codecs/vq_images_unquantized_with_clustering_v2.mp4" class="img-fluid rounded z-depth-1"  controls=true muted=true autoplay=true loop=true %}
<div class="caption">
  Quantizing by fitting a clustering on top of the autoencoder
</div>

You can see the reconstructions of the cluster centers getting refined over time.

Next, we’ll make the encoder and decoder themselves better at handling quantized embeddings during training, because currently, we’re just fitting the clustering on top of an autoencoder that is not “aware” it’s being quantized. We’d like the autoencoder to adapt to the quantization as we train it. Currently, we’re doing this:

```python
x = get_batch()
z = encoder(x)

x_reconstructed = decoder(z)

loss = reconstruction_loss(x, x_reconstructed)
```

Instead of feeding the unquantized embedding into the decoder, we’ll first move it to the closest cluster:

```python
x = get_batch()
z = encoder(x)

z_quantized = to_nearest_cluster(z)     # 👈
x_reconstructed = decoder(z_quantized)  # 👈

loss = reconstruction_loss(x, x_reconstructed)
```

There is a snag: if we do this, we won’t be able to train the autoencoder any more, because the quantization operation is not differentiable, meaning there is no gradient flowing from the loss to the weights of the encoder. Essentially, we’re no longer able to answer the question: “if I want the loss to decrease a bit, in which direction should I nudge the encoder’s weights?”

We’ll fix this problem by pretending it doesn’t exist. Yes, really. We’ll think of `z_quantized` as `z` moved by an arbitrary vector that doesn’t affect the gradient. That will make the gradient of `z` equal to that of `z_quantized`, which is why this is also known as the _straight-through estimator_ of the gradient.

```python
x = get_batch()
z = encoder(x)

residual = z - to_nearest_cluster(z)
# .detach() means "forget that this needs a gradient"
z_quantized = z - residual.detach()
x_reconstructed = decoder(z_quantized)

loss = reconstruction_loss(x, x_reconstructed)
```

In the forward pass, `z_quantized` is set to the same value as before, but importantly, the gradient of `z` is now equal to that of `z_quantized` rather than just being 0 because of the non-differentiable `to_nearest_cluster(z)` operation.

There is a price to pay for this lie. When training, the encoder’s weights will be updated to improve the reconstruction loss, but they’re updated as if the quantization didn’t happen, so they won’t move in the optimal direction. But as long as the embeddings stick close to their cluster centers, the gradient direction will still be mostly correct.

We can actually encourage the encoder to make embeddings that are easily quantizable by adding a _commitment loss_: a penalty for each point based on how far it is from its cluster center. The gradient of this loss will push the points closer to their cluster centers.

By quantizing at training time and adding a commitment loss, it’s no longer just a clustering being fit on top of the embeddings. The model itself is trained to be good for quantization.

{% include video.liquid path="assets/img/2026-04-27-neural-audio-codecs/vq_images_balanced_v2.mp4" class="img-fluid rounded z-depth-1 w-full max-w-72" controls=true muted=true autoplay=true loop=true %}
<div class="caption">
  An autoencoder trained explicitly to be easy to quantize
</div>

You’ll notice that the training dynamics look different: the commitment loss adds a certain “stiffness” that doesn’t allow the embeddings to move around as easily.

Here’s what the reconstructions look like when we use the quantized representations:

{% include figure.liquid path="assets/img/2026-04-27-neural-audio-codecs/rvq-1-level-v4.png" class="img-fluid rounded z-depth-1 bg-black" %}

Notice how the first two images are reconstructed to _exactly_ the same image. That’s simply because their embeddings got assigned to the same cluster and therefore quantized to the same value.

The model described here is known as a VQ-VAE <d-cite key="DBLP:journals/corr/abs-1711-00937" />: a vector-quantized variational autoencoder. The word “variational” here is just a vestigial leftover that doesn’t mean anything anymore <d-cite key="dieleman2025latents" />.

## Residual vector quantization

To improve the reconstruction fidelity, we can just increase the number of cluster centers. But keeping track of too many centers can get prohibitively expensive in terms of compute and memory required, so we’ll do a clever trick: if we want $2^{20}$ (~1M) possible values, we won’t create $2^{20}$ clusters directly. Instead, we’ll use two separate quantizers with $2^{10}=1024$ clusters and combine their result. Each embedding will then be quantized to a tuple of two integers in [0..1023], yielding $2^{20}$ possible combinations.

Ok, but how? Well, recall the `residual` variable we used in the straight-through estimator, defined as `z - to_nearest_cluster(z)` the shift from the quantized embedding to the unquantized one. It represents the part of the original vector `z` that we didn’t manage to take into account when quantizing to `to_nearest_cluster(z)`.

So for each embedding in the batch, we have a corresponding residual vector. The solution is obvious: we’ll quantize these residuals exactly the same way we did with the original embeddings, by training another vector quantizer.

This time, the 2D positions for a single quantizer don’t define images because we need to combine the two quantizers, so we’ll just visualize everything as dots:

{% include video.liquid path="assets/img/2026-04-27-neural-audio-codecs/rvq_fmnist.mp4" class="img-fluid rounded z-depth-1" controls=true muted=true autoplay=true loop=true %}
<div class="caption">
  Two-level quantization by fitting a quantizer on top of the
  &ldquo;residuals&rdquo;, aka the errors of the first quantizer
</div>

Each image is then represented as the index of the cluster of the embedding and that of the residual. Let’s try to reconstruct a few images with this two-level quantizer:

{% include figure.liquid path="assets/img/2026-04-27-neural-audio-codecs/rvq-2-level-v4.png" class="img-fluid rounded z-depth-1 bg-black" %}
<div class="caption">
Original images (top), one-level reconstruction (middle), two-level reconstruction (bottom). These images are encoded as (4, 3), (4, 5), (16, 21), and (30, 3).
</div>

The reconstructions of the first two images are similar, but no longer the exact same: the first image is represented as (4, 3) and the second as (4, 5). In other words, they share the same token for the first level, but differ in how the residual is quantized. The differences are quite subtle, so here’s a comparison between the one-level and two-level reconstructions:

{% include figure.liquid path="assets/img/2026-04-27-neural-audio-codecs/rvq-2-level-diff-v3.png" class="img-fluid rounded z-depth-1"  %}
<div class="caption">
Difference between one-level and two-level reconstructions
</div>

I’d like to emphasize that the second quantization level makes modifications to the embedding, not the output pixels directly. This can be seen by the fact that the leftmost and rightmost image are encoded as (4, 3) and (30, 3) respectively. So they have the same residual code, 3, but it modifies the two reconstructed images in different ways.

Clearly, the reconstructions are still not very accurate. The upper bound on the quality is the reconstruction from unquantized embeddings, so if your autoencoder is bad (and ours is), improving the quantization won’t save you.

We’ll stop here, but a natural extension to this idea is to go beyond two levels. Just take the residuals of the two-level reconstruction and quantize those, and so on. This generalized Residual Vector Quantization algorithm looks like this:

```python
def rvq_quantize(z):
    residual = z
    codes = []

    for level in range(levels):
        quantized, cluster_i = to_nearest_cluster(level, residual)
        residual -= quantized
        codes.append(cluster_i)

    return codes
```

Residual vector quantization was first applied to neural audio codecs in SoundStream <d-cite key="DBLP:journals/taslp/ZeghidourLOST22" />, but the idea has been around since the 80s <d-cite key="multiplestage82" />.

## Now let’s tokenize audio

Applying RVQ to audio is fairly straightforward. As our autoencoder, we’ll use a convolutional neural network (CNN) similar to [what Jukebox uses](https://github.com/openai/jukebox/blob/08efbbc1d4ed1a3cef96e08a931944c8b4d63bb3/jukebox/vqvae/encdec.py) <d-cite key="DBLP:journals/corr/abs-2005-00341" />. The details of the architecture aren’t too important here. What’s important is that it’s a network that takes an audio with $t$ samples and converts it to a vector of shape $(\frac{t}{128}, 32)$. In other words, it downsamples by a factor of 128 and gives us 32-dimensional float representations. The decoder then takes the $(\frac{t}{128}, 32)$ embeddings and decodes them back into $t$ samples.

```python
audio = get_batch()               # shape: [B, T]
z = encoder(audio)                # shape: [B, T/128, 32]
audio_reconstructed = decoder(z)  # shape: [B, T]
```

As before, we’ll add an RVQ after the encoder. The only difference from the image case is that for each audio sample, we have $\frac{t}{128}$ embedding vectors, not just a single one as we did for images. We just quantize these independently (even though the encoder “sees” more audio than what corresponds to that one vector). During training, we also have a batch dimension, so our model now looks like this:

```python
audio = get_batch()                         # [B, T]
z = encoder(audio)                          # [B, T/128, 32]

# Combine the batch and time dimensions
z = rearrange(                              # [B*T/128, 32]
    z, "b t_emb d -> (b t_emb) d"
)

codes = rvq_quantize(z)           # integers, [B*T/128, levels]
z_quantized = codes_to_embeddings(codes)    # [B*T/128, 32]
z_quantized = rearrange(                    # [B, T/128, 32]
    z_quantized, "(b t_emb) d -> b t_emb d"
)

audio_reconstructed = decoder(z_quantized)  # [B, T]
```

{% include video.liquid path="assets/img/2026-04-27-neural-audio-codecs/codec-with-rvq.mp4" class="img-fluid rounded z-depth-1" controls=true muted=true autoplay=true loop=true %}

The last missing piece before we can train our first neural audio codec is a loss function. There’s a whole rabbit hole we could go into about which one to choose, but we’ll avoid it and just use a very simple one. We’ll compute the log amplitude spectrogram of the original and reconstructed audio, and take their difference. The loss is the mean square of this difference between spectrograms.

To make it harder for the model to overfit to this loss, we take the spectrogram with three different parameters for the short-time Fourier transform, and let our loss be the mean between the three sub-losses. This is called the _multi-scale spectral loss_.

{% include figure.liquid path="assets/img/2026-04-27-neural-audio-codecs/image-2.png" class="img-fluid rounded z-depth-1" %}

<div class="caption">
Image from Evan Radkoff’s excellent blog
post <d-cite key="loss_functions_audio_ml" /> about loss
functions in audio ML. Check it out if you want to go down the loss function
rabbit hole.
</div>

Finally, let’s train some codecs! We’ll look at how varying the number of RVQ levels affects the reconstruction quality. As we expected, increasing the number of levels helps, decreasing the spectral loss:
{% include figure.liquid path="assets/img/2026-04-27-neural-audio-codecs/image-3.png" class="img-fluid rounded z-depth-1 max-w-104" %}

Let’s hear what the codecs sound like. We’ll use the three codecs to reconstruct this audio from the Expresso dataset <d-cite key="DBLP:journals/corr/abs-2308-05725" />:

{% include audio.liquid path="assets/img/2026-04-27-neural-audio-codecs/orig_audio.wav" controls=true class="audio-sample" %}

And the reconstructions:

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    <div class="font-weight-semibold mb-1">4 RVQ levels</div>
    {% include audio.liquid path="assets/img/2026-04-27-neural-audio-codecs/recon_4_rvq.wav" controls=true class="audio-sample" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    <div class="font-weight-semibold mb-1">8 RVQ levels</div>
    {% include audio.liquid path="assets/img/2026-04-27-neural-audio-codecs/recon_8_rvq.wav" controls=true class="audio-sample" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    <div class="font-weight-semibold mb-1">16 RVQ levels</div>
    {% include audio.liquid path="assets/img/2026-04-27-neural-audio-codecs/recon_16_rvq.wav" controls=true class="audio-sample" %}
  </div>
</div>

Clearly, the audio gets better as we add more RVQ levels.

Even with 16 levels, there is some crackling, the audio sounds muffled, and there is a constant high-pitched noise. Later we’ll discuss how we could improve the codec further, but for demonstration purposes, this will do.

## Why care about audio

So now we have a neural audio codec: we can turn audio into LLM-friendly tokens and back. Codec just means a tokenizer for audio, but we say _codec_ because that’s the term used for classic compression like MP3. I’ll be using codec and tokenizer interchangeably.

Let’s come back to what we wanted to do in the first place: modeling audio. Specifically, we’ll make a model that can take an audio prefix and generate a plausible continuation for it.

Just as a reminder, we want to train good audio LLMs so that we have models that understand and produce speech natively, understanding emotion, emphasis, and so on. They could also be fine-tuned into text-to-speech, speech-to-text, or speech translation models <d-cite key="DBLP:journals/corr/abs-2502-03382" />, among others.

So now that you’re convinced that audio LLMs are the path to AGI, let’s train a few.

For our dataset, we’ll use Libri-Light <d-cite key="DBLP:conf/icassp/KahnRZKXMKLCFLS20" />, like we did for our sample-by-sample model earlier. This time we’ll use 10000h of audio instead of 1000h. It’s a dataset of public-domain audiobooks, so if we have a good model for it, maybe we’ll be able to generate more stories. (Don’t get your hopes up too much.) All we need to do is to convert the audio dataset into a sequence of discrete tokens so that we can feed it into an LLM.

## Dealing with multiple levels

We’ll do that using our 8-level RVQ codec. From an audio with $t$ samples, we’ll get an array of tokens of shape $(\frac{t}{128}, 8)$. But now there’s an issue: how to deal with the fact that for each time step, there’s not one but eight tokens? This is not a problem we have to deal with in text LLMs, where we have a single sequence of tokens.

We’ll do the simplest thing possible and just flatten the array into 1D of shape $(\frac{t}{128} \cdot 8)$, and have our LLM predict the eight levels in separate time steps.

{% include video.liquid path="assets/img/2026-04-27-neural-audio-codecs/flatten-rvq.mp4" class="img-fluid rounded z-depth-1" controls=true muted=true autoplay=true loop=true %}
<div class="caption">
  Flattening a three-level RVQ to allow it to be fed into a language model
</div>

The big disadvantage is that we lose some of our temporal compression. We downsampled the audio 128x, but now we’re inflating it 8x again by flattening the levels. That makes inference less efficient, and possibly worse quality because the effective context size decreases. We'll be using the 8 RVQ codec rather than the 16 RVQ one to avoid making the compression even worse.

You could also predict all RVQ levels for a single step at once (”parallel pattern”), but it also makes things harder for the model because it has to decide on all levels at once. There are a bunch of other schemes people have tried to balance compression and quality. Here are a few tried out in MusicGen:

{% include figure.liquid path="assets/img/2026-04-27-neural-audio-codecs/image-4.png" class="img-fluid rounded z-depth-1" %}

<div class="caption">
Figure taken from the MusicGen paper <d-cite key="DBLP:journals/corr/abs-2306-05284" />.
</div>

Interestingly, as of 2025, there is no single solution that “won”: every paper does something different, and the schemes can get quite involved. Just look at this diagram from MiMo-Audio <d-cite key="mimoaudio" />, a model released in September 2025:

{% include figure.liquid path="assets/img/2026-04-27-neural-audio-codecs/image-5.png" class="img-fluid rounded z-depth-1"  %}
<div class="caption">
Ways to deal with multiple RVQ levels can get quite involved
</div>

## Finally, let's train

Time to finally train a codec-wrapped language model! As I’ve mentioned, our code is based on Andrej Karpathy’s [nanoGPT codebase](https://github.com/karpathy/nanoGPT) for training text LLMs. We just need to modify it to accept audio as input. But that’s easy, because LLMs don’t care about what kind of tokens you’re feeding in – it’s all just numbers. Once we’ve tokenized the dataset and flattened it into a 1D sequence, we’re good to go. Tokenized this way, our 10000 hours of audio take up 134 GB. For comparison, storing this much data as uncompressed audio would take over 1 TB.

We’re going to use the exact same model architecture and hyperparameters as for the sample-by-sample model: the only difference is in the tokenization. We also have a 10x bigger dataset, but the sample-by-sample model can’t even fit the dataset with 1k hours, so more data wouldn’t save it.

I trained the model on 8 H100s for about 5 days. To get some samples, I decided to prompt the model with a sample of Libri-Light reading of two lines from Michael Field’s poem July <d-cite key="michael_field" />. (As I learned when working on this, Michael Field is a pen name of Katherine Harris and Edith Emma Cooper.) Let’s see what kind of poetry we can get from our model:

{% include audio.liquid path="assets/img/2026-04-27-neural-audio-codecs/20251002_123351_4.wav" controls=true class="audio-sample" %}

There are some signs of life, but we don’t have a poet yet. It sounds like somebody speaking behind a curtain. You can’t really make out what it’s saying, but the intonation is there: it sounds like somebody reading from a book, which is indeed what the model was trained on.

It also maintains a coherent voice, until it decides for the last few seconds to switch to a different one. That is also consistent with the data: we sample the training data from a concatenation of all the audiobooks chopped up into segments and mixed together, so the model does encounter boundaries between different speakers.

## How far can a codec get us?

Our codec was deliberately simplistic, which explains why the results aren't great—but there's been a good amount of research on neural audio codecs in the last four years that we could leverage.
We won’t implement all the improvements here, but instead we’ll look at what happens when we use [Mimi](https://huggingface.co/kyutai/mimi) as the tokenizer.

Mimi is a modern neural audio codec built for the audio language model Moshi <d-cite key="DBLP:journals/corr/abs-2410-00037" />. It’s since been used as the tokenizer for other models as well, like Sesame CSM <d-cite key="sesame_uncanny_valley_voice" />, VoXtream <d-cite key="DBLP:journals/corr/abs-2509-15969" />, and LFM2-Audio <d-cite key="lfm2_audio" /> ([GitHub](https://github.com/Liquid4All/liquid-audio)).

Unsurprisingly, Mimi sounds a lot better than the homemade codec we trained earlier.

Instead of the multi-scale spectral loss, Mimi uses an adversarial loss, like a GAN. There’s a discriminator network that tries to classify audios as being original or reconstructed by the codec, and the goal of the codec is to fool this discriminator.

Another improvement Mimi adds is using RVQ dropout: it uses 32 RVQ levels but during training, the reconstruction is sometimes randomly truncated to a lower number of levels. That allows us to run Mimi for a lower number of RVQ levels at inference time and still get decent results, because it doesn’t rely on all levels being present. For our codec, we had to train separately.

Let’s hear our example audio reconstructed with Mimi:

Original

{% include audio.liquid path="assets/img/2026-04-27-neural-audio-codecs/original_24kHz.wav" controls=true class="audio-sample" %}

<div class="row mt-3">
  <div class="col-sm w-100">
    4 RVQ levels
    {% include audio.liquid path="assets/img/2026-04-27-neural-audio-codecs/mimi_4_rvq_recon.wav" controls=true class="audio-sample" %}
  </div>
  <div class="col-sm w-100">
    8 RVQ levels
    {% include audio.liquid path="assets/img/2026-04-27-neural-audio-codecs/mimi_8_rvq_24kHz.wav" controls=true class="audio-sample" %}
  </div>
</div>

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    <div>16 RVQ levels</div>
    {% include audio.liquid path="assets/img/2026-04-27-neural-audio-codecs/mimi_16_rvq_recon.wav" controls=true class="audio-sample" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    <div>32 RVQ levels</div>
    {% include audio.liquid path="assets/img/2026-04-27-neural-audio-codecs/mimi_24kHz.wav" controls=true class="audio-sample" %}
  </div>
</div>

For our purposes, a variant with fewer levels might have the advantage of being easier to model because it’s more compressed. Let’s train models with 8- and 32-level Mimi and compare the results.

I trained the exact same model architecture as before, the only thing that changes is the tokenizer. It’s 10k hours from Libri-Light as the dataset, just like when we used our simple codec. Mimi has a sample rate of 24 kHz but Libri-Light uses 16 kHz, which puts a cap on how good it can sound, since we lose the higher frequencies of the audio.

Mimi downsamples the audio a lot more aggressively, too: its sample rate is 12.5 frames per second, whereas we used 125 frames per second for our codec – 10x higher! This means the dataset is also smaller on disk. With our codec, it took 134 GB, but for Mimi it’s “just” 54 GB.

Here’s a poem generated with the model trained on Mimi-tokenized data. I prompted it with two lines from the poem, as before:

{% include audio.liquid path="assets/img/2026-04-27-neural-audio-codecs/20251002_115006_2.wav" controls=true class="audio-sample" %}

Here is my best attempt at a transcription:

> _When the grass is gone<br/>
> And corn still grassy;_<br/>
> Illness worried in the fur<br/>
> this and pelan in stones<br/>
> during the turan’s ciscerey<br/>
> headforths nepet Paul Twain.<br/>
> He _sees_ zin in them.<br/>

A tad too surrealist for my taste, but maybe Lewis Carroll would like it.

## Semantic tokens

I have a confession to make: I lied to you just now. But just a bit, and for didactic purposes. In fact, the model above was trained on audio from a 31-level Mimi, where I omitted the very first level, which contains the “semantic token”.

The role of this token is to represent semantic information of the audio, without necessarily aiding reconstruction. I won’t go into how these work, but in one sentence, Mimi’s semantic tokens are distilled from WavLM <d-cite key="DBLP:journals/corr/abs-2110-13900" />, which you can think of as a BERT <d-cite key="DBLP:journals/corr/abs-1810-04805" /> for speech.

To get a feeling for what information semantic tokens encode, let’s take this example audio, passed through Mimi:

{% include audio.liquid path="assets/img/2026-04-27-neural-audio-codecs/original.wav" controls=true class="audio-sample" %}

Now let’s train a language model trained on the full Mimi, including semantic tokens. We’re going to run the model in a way where we keep the semantic tokens from the original audio but we discard the others, and let the model predict them. That means the information from the semantic tokens is fixed (”teacher-forced”), but the model is free to decide the others according to what continuations it finds plausible.

{% include video.liquid path="assets/img/2026-04-27-neural-audio-codecs/regenerate-with-semantic.mp4" class="img-fluid rounded z-depth-1" controls=true muted=true autoplay=true loop=true %}
<div class="caption">
  We can get an idea of what information is contained in semantic tokens by
  keeping them fixed and letting the model regenerate the rest.
</div>

Listen to two different reconstructions we obtain this way:

{% include audio.liquid path="assets/img/2026-04-27-neural-audio-codecs/regenerate-1.wav" controls=true class="audio-sample" %}

{% include audio.liquid path="assets/img/2026-04-27-neural-audio-codecs/regenerate-2.wav" controls=true class="audio-sample" %}

The voice is completely different, but it’s saying the same thing! This means the semantic tokens encode what the person is saying, but are invariant to the voice. That’s useful because it helps the model focus on _what_ to say, not _how_ to say it. In that regard, they’re closer to text tokens, which also don’t contain information about the voice, intonation, timing, or emotion.

## Making poetry semantic

Now let’s take the model trained on semantic Mimi and ask it to complete the poem:

{% include audio.liquid path="assets/img/2026-04-27-neural-audio-codecs/20251002_115255_0.wav" controls=true class="audio-sample" %}

> _When grass is gone<br/>
> and corn still grassy;_<br/>
> from the man was nothing moan.<br/>
> The low death and heart<br/>
> She came _fyde_ wood.<br/>
> A finteriest, a fall,<br/>
> all them.<br/>

It still makes up words and the sentences are not too coherent, but clearly, the proportion of real words is much higher; the model is “more semantic”. The acoustic quality is the same, which is what we’d expect.

Let’s listen to a second poem:

{% include audio.liquid path="assets/img/2026-04-27-neural-audio-codecs/20251002_115255_2.wav" controls=true class="audio-sample" %}

> _When grass is gone<br/>
> and corn still grassy;_<br/>
> hope won and she<br/>
> who is just a night in Tatan<br/>
> in doe ock-ohm?<br/>
> the whom?<br/>

Indeed, the whom?

## Semantic–acoustic tradeoff

We can sacrifice some acoustic quality to improve the semantics by reducing the number of RVQ levels. Let’s do 8. That way, we get higher audio compression, and a proportionally higher part of the loss comes from the semantic token, since now it’s $\frac{1}{8}$ tokens and not just $\frac{1}{32}$.

One of the first things I noticed about this model is that it learned to memorize the Librivox notice, so it sometimes generates things like:

{% include audio.liquid path="assets/img/2026-04-27-neural-audio-codecs/20251002_121528_3_librivox_intro_trim.wav" controls=true class="audio-sample" %}

> Chapter 6 of The Founday, by R. Auclair.<br/>
> This is a Librivox recording. All Librivox recordings are in the public domain. For information, or to volunteer, please visit librivox.org.<br/>
> Reading by: Kelvert

Repeating the training data is generally not what you want, but in our case it’s a great sign of life, because the previous models couldn’t even manage that. It also makes up the book, author, and reader, so there is still novelty here.

Now let’s try to make some more poetry:

{% include audio.liquid path="assets/img/2026-04-27-neural-audio-codecs/20251002_120917_0_mimi8_temp08_trim.wav" controls=true class="audio-sample" %}

> _When grass is gone<br/>
> and corn still grassy;_<br/>
> When so we could say<br/>
> that in fairy interesting wife<br/>
> who lay there and gone<br/>
> that save the rosy light of life<br/>
> Jay Dien, the antique mollity<br/>
> and a mollity the beast of gray failed summon<br/>
>
> end of poem.
>
> This recording is in the public domain.
>
> [different voice]<br/>
> So we have formed a float that sent in would rattle down. The piece of opportunity reading and assimila—

This is great. There are several signs of the model being better than the previous ones. I love that it makes up the word “mollity” and then repeats it in the next line. Also, it realizes that it’s reciting a poem and ends the section with “end of poem”. Then it decides it’s the end of the chapter/section and it ends with the “This recording is in the public domain.” disclaimer. After that, it changes the voice and continues talking. That makes sense, since the clips from various audiobooks are just shuffled and concatenated during training, so here the model simulated a clip boundary.

We might get even better results by weighing the loss of the semantic tokens higher than the acoustic tokens, to make the model focus more on the meaning than the sound – in fact, Moshi uses a semantic loss factor of 100x! But we have to stop somewhere.

## Conclusion

We’ve managed to use neural audio codecs to make an audio language model that generates somewhat coherent speech. Obviously, that’s not where the state of the art is in 2025 (and we’re not trying to reach it here) but keep in mind that by using the _exact same model_ without neural audio codecs gives us this:

{% include audio.liquid path="assets/img/2026-04-27-neural-audio-codecs/20250925_140600_3.wav" controls=true class="audio-sample" %}

Of course, still a long way to go to match text models! Currently, there seems to be a trade-off between speech understanding and reasoning abilities. At the beginning, I mentioned that the speech-native models
(Gemini 2.5 <d-cite key="DBLP:journals/corr/abs-2507-06261" /> <d-cite key="gemini_2_5_native_audio" />,
ChatGPT’s Advanced Voice Mode <d-cite key="DBLP:journals/corr/abs-2410-21276" />,
Qwen <d-cite key="DBLP:journals/corr/abs-2509-17765" />,
Moshi <d-cite key="DBLP:journals/corr/abs-2410-00037" />)
aren’t able to tell you whether you’re speaking in a high or low voice, despite the fact that they’re trained to natively understand audio. This is likely because they’re trained on a lot of data generated synthetically with text-to-speech and/or because understanding the tone of the voice (apparently) doesn’t help the models make more accurate predictions.

Audio models still lag behind text LLMs. But why? To me, this mysterious unsolved “modality gap” makes audio ML an exciting field to work on.
