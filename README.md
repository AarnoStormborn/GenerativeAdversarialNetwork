# Problem Statement

<p style="font-size:25px">Generating Brain Tumor MRI images for Data Augmentation using Generative Adversarial Networks</p>

# About Brain Tumor

<p style="font-size:20px">A brain tumor is a collection, or mass, of abnormal cells in your brain. Your skull, which encloses your brain, is very rigid. Any growth inside such a restricted space can cause problems.
Brain tumors can be cancerous (malignant) or noncancerous (benign). When benign or malignant tumors grow, they can cause the pressure inside your skull to increase. This can cause brain damage, and it can be life-threatening.
Brain tumors are categorized as primary or secondary:
<ul>
    <li style="font-size:20px">A primary brain tumor originates in your brain. Many primary brain tumors are benign.</li>
    <li style="font-size:20px">A secondary brain tumor, also known as a metastatic brain tumor, occurs when cancer cells spread to your brain from another organ, such as lung or breast. </li>
</ul>
</p>

![healthline](https://i0.wp.com/post.healthline.com/wp-content/uploads/2022/02/2009199_Understanding-Brain-Tumors-01.jpg?w=1155&h=1887)
<br>

# How is Brain Tumor diagnosed?
<br>

![Brain Tumor](https://qph.cf2.quoracdn.net/main-qimg-3cd4287f29cd7fdc7f68ab223107bb64-pjlq)
<br>

## Magnetic Resonance Imaging (MRI)
<p>
    <ul>
        <li  style="font-size:20px">An MRI uses magnetic fields to produce detailed images of the body.<br> MRI can be used to measure the tumor’s size. A special dye called a contrast medium is given before the scan to create a clearer picture.</li> 
        <li  style="font-size:20px">This dye can be injected into a patient’s vein or given as a pill or liquid to swallow.<br> MRIs create more detailed pictures than CT scans and are the preferred way to diagnose a brain tumor.</li> 
        <li  style="font-size:20px">The MRI may be of the brain, spinal cord, or both, depending on the type of tumor suspected and the likelihood that it will spread in the CNS.</li>
        <li  style="font-size:20px">There are different types of MRI. The results of a neuro-examination, done by the internist or neurologist, helps determine which type of MRI to use.</li>
    </ul>
</p>


# What do the Numbers Say?

<ul>
    <li style="font-size:20px">In India, every year, 40,000 - 50,000 patients are diagnosed with a brain tumor. 20 percent of them are children</li>
    <li style="font-size:20px">At the current population level of the country (1.417 billion), this means only <b>0.0035 percent</b> are diagnosed with Brain Tumor!</li>
    <li style="font-size:20px">Let's assume that all MRI scans produce 100% accurate results. This would mean that for every 10,000 MRI scans, we only get <b>35 samples</b> showing Brain Tumor versus many more that don't</li>
    <li style="font-size:20px">This, combined with other problems in accessing Medical data, would lead to Machine Learning problems such as <b>Class Imbalance</b> and <b>Bias</b></li>
</ul>

<p style="font-size:15px">Source: https://health.economictimes.indiatimes.com/news/diagnostics/brain-tumors-death-on-diagnosis/88090467</p>


# A Solution - Generative Modelling

<p style="font-size:20px">Generative models, or deep generative models, are a class of deep learning models that learn the underlying data distribution from the sample. These models can be used to reduce data into its fundamental properties, or to generate new samples of data with new and varied properties</p>

# Generative Adversarial Networks

<p style="font-size:20px">Generative adversarial networks are implicit likelihood models that generate data samples from the statistical distribution of the data. They’re used to copy variations within the dataset. They use a combination of two networks: generator and discriminator.</p>
<br>

![GAN](https://miro.medium.com/max/720/1*9jwIuW0KPi3THIvoYg9BUQ.png)

## <u> The Generator: </u>
<p style="font-size:20px">A generator network takes a random normal distribution (z), and outputs a generated sample that’s close to the original distribution.</p>

## <u> The Discriminator: </u>
<p style="font-size:20px">A discriminator tries to evaluate the output generated by the generator with the original sample, and outputs a value between 0 and 1. If the value is close to 0, then the generated sample is fake, and if the value is close to 1 then the generated sample is real.</p>

## <u> What the Entire thing looks like: </u>

<br>

![DC-GAN](https://camo.githubusercontent.com/45e147fc9dfcf6a8e5df2c9b985078258b9974e3/68747470733a2f2f63646e2d696d616765732d312e6d656469756d2e636f6d2f6d61782f313030302f312a33394e6e6e695f6e685044614c7539416e544c6f57772e706e67)

## <u> How do GANs work ? </u>

<p style="font-size:20px">A random normal distribution is fed into the generator. The generator then outputs a random distribution, since it doesn’t have a reference point. <br>
Meanwhile, an actual sample, or ground truth, is fed into the discriminator. The discriminator learns the distribution of the actual sample. When the generated sample from the generator is fed into the discriminator, it evaluates the distribution.<br>
If the distribution of the generated sample is close to the original sample, then the discriminator outputs a value close to ‘1’ = real. If both the distribution doesn’t match or they aren’t even close to each other, then the discriminator outputs a value close to ‘0’ = fake.</p>

## <u> The Minimax setting </u>

<br>

![MiniMax](https://media.geeksforgeeks.org/wp-content/uploads/g22-1.png)

<p style="font-size:20px">The answer lies in the loss function or the value function; it measures the distance between the distribution of the data generated and the distribution of the real data. Both the generator and the discriminator have their own loss functions. The generator tries to minimize the loss function while the discriminator tries to maximize.</p>
