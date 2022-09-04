""":cvar
A machine learning paper is a scientific paper that details findings of a research group on a specific area.
The contents of a machine learning reasearch paper can vary from paper to paper but they generally follow the structure:

Section/Contents:
Abstract: An overview/summary of the paper's main findings/contributions.
Introduction: What's the paper's main problem and details of previous methods used to try and solve it.
 Method: How did the reseachers go about conductiong thier research? e.g. what model(s), data source(s), training setups, and
 were used...
 Results: What are the outcomes of the paper? If a new type of model or training setup was used, how did the results is
 of findings compare to previous works? (this is where experiment tracing comes in handy)
 Conclusions: WHat are teh limitations of the suggested methods? WHat are some next steps for the reserch community?
 Reference: What resources/other papers did the reseachers look at to build their own body of work?
 Appendix: Are there any extra resources/findings to look a t that weren't included in any of the above sections?

"""

print(f"Why replicating a machine learning paper? ::\n"
      f"\nA machine learning reserch paper is often a presentation of months of work and experiments done by some of the best work"
      f"\nmachine learning teams in thw world condensed into a few pages of text..."
      f"\nAnd if these experiments lead to better results in an area related to the probelm you're working on,"
      f"it'd be nice to them out... \nAlso, replicating the work of others is a fantastic way to practice your skills.. "
      f"\n \n \n Machine learning engineer: "
      f"\n1. Download a paper "
      f"\n2. Implement it.."
      f"\n3. Keep doing this until you have skills .."
      f"\n by George Hots, a comm.ai founder,  a self-driving car company an dliverstreams maching  learning coding "
      f"\n on Twitch and those videos get posted in full to Youtube .. \n What we're going to cover  Rather than talk about a replicating a paper, we re going to get hands-on and actually replicate a paper. The process for replicating all papers will be slightly different but by seeing what it's like to do one, we'll get the momentum to do more.")

"""
More specifically, we're going to be replicating the machine learning research paper An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT paper) with PyTorch.

The Transformer neural network architecture was originally introduced in the machine learning research paper Attention is all you need.

And the original Transformer architecture was designed to work on one-dimensional (1D) sequences of text.
"""
print("A Transformer architecture is generally considered to be any neural network that uses the attention mechanism) as its primary learning layer. Similar to a how a convolutional neural network (CNN) uses convolutions as its primary learning layer.")


"""
Like the name suggests, the Vision Transformer (ViT) architecture was designed to adapt the original Transformer architecture to vision problem(s) (classification being the first and since many others have followed).

The original Vision Transformer has been through several iterations over the past couple of years, however, we're going to focus on replicating the original, otherwise known as the "vanilla Vision Transformer". Because if you can recreate the original, you can adapt to the others.

We're going to be focusing on building the ViT architecture as per the original ViT paper and applying it to FoodVision Mini.
"""


print(f"1. Getting setup: reuse our existing code")
print(f"2. Get data: Look at our dataset")
print(f"3. Create Datasets and DataLoaders: using data_setup.py script")
print(f"4. Replicating the ViT paper: an overview: replicating a machine learning research paper can be bit a fair challenge, so before we jump in, let's break the ViT paper down into smaller chunks, so we can replicate the paper chunk by chunk")
print(f"5. Equation 1: The Patch Embedding - The ViT architecture is comprised of four main equations, teh first being the path and poition encoding/embedding. By turning an image into a sequence of learnable patches.")
print(f"6. Equation 2: Multi Head Self Attention (MSA) - the self-attention/multi-head self-attention (MSA) mechanism is at the heart of every Transformer architecture, including the ViT architecture, let's create an MSA block using PyTorch's in-built layers.")
print(f"7. Equation 3: Multilayer Perceptron (MLP) - The ViT architecture uses a multilayer perceptron as part of its Transformer Encoder and for its output layer. Let'ws start by creating an MLP for the Transformer Encoder.")
print(f"8. Creating the transformer Encode: A transformer Encoder is typically comprised of alternating layers of MSA (equation 2) and MLP (equation 3) jointed together via residual connections. Let's create one by stacking the layers we created in sections 6 & 7 on top each other.")
print(f"9. Putting it all together to create ViT - we have got all the pieces of puzzsle to create teh ViT architecture, let's put them all together into a single class we can call as our model.")
print(f"10. Setting up training code for our ViT model: Training our custom ViT implementation is similar to all of the other models we have trained previously. And thenks to our train() functions in engine.py script we can start training with few line of codes")
print(f"11. Using a pretrained ViT from torchvision.models - Training a large model like ViT usually takes a fair amount of data. Since we're only working with a small amount pf dataset, let's see fi we can leverage the power of transfer learning to  imporve our perforemance")
print(f"12. Make predictions on a custom image: The magic of machine learning is seeing it work on you own data, let's take our bet performance model an dput FoodVision Minin to the test on the infamous datasets.")
print(f"*" * 80)
print(f"\n Note: Despite the fact we're going to be focused on replicating the ViT paper, avoid getting too bogged down on a particular paper as newer better methods will often come along, quickly, so the skill should be to remain curious whilst building the fundamental skills of turning math and words on a page into working code.")



