
# Implementation of the Fast Fourier and Discrete Cosine Transform

Author: Pramith Devulapalli

Alternate Version: A corresponding python notebook is in the Jupyter Notebooks folder

The Discrete Cosine Transform (DCT) is a version of the Discrete Fourier Transform (DFT) that aims to represent an arbitrary signal in terms of purely cosine functions. Through the application of the DCT, the spectral representation of the image can be acquired and the underlying frequencies in the image can be analyzed. 

The concept of the DCT lies in its basis functions. The 1-D DCT
<p align="center"><img alt="$$ X_k = \sum_{n=0}^{N-1} x_n cos[\frac{\pi}{N} (n + \frac{1}{2}) k] $$" src="svgs/17ecf3d09b4ee9c152779f3866b8dcdc.svg" align="middle" width="252.15488pt" height="57.706026pt"/></p> where <p align="center"><img alt="$$ k = 0, \ldots, N-1$$" src="svgs/7bbefffecadf4b3a72f9d5a1e52513ce.svg" align="middle" width="144.305438pt" height="17.711368pt"/></p>
constructs an arbitrary signal through cosines of k varying frequencies.

The DCT is used in computer vision for a different number of reasons. The first (0th) coefficient of the DCT is the mean of the values being transformed. This is crucial because the first coefficient represents the average tone of the image pixels and represents important information (a DST doesn't have this property). Additionally, the boundary conditions implied by the DCT differ from the DFT. The DFT implies a periodic extension of the signal whereas the DCT implies an even extension of the signal (symmetric). In imaging applications, the DCT is more conducive than the DFT because the DCT is able to represent most of the signal information in a smaller amount of terms than that of the DFT.  


```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import scipy as sc
import scipy.fftpack
import scipy.misc as misc
import cv2
import math
```


```python
img2 = cv2.imread('DCT.png', 0)
```

### DCT Basis Functions

Greyscale images are represented as an NxM matrix of values that range from 0-255. In order to perform the DCT, the two dimensional DCT needs to be performed on the image to extract the spectral representation of 2-d matrix. Fortunately, the 2-D DCT is simply an extension of the one dimensional DCT. Below is the 2-D DCT:


```python
plt.figure(figsize=(15,15))
plt.imshow(img2, cmap='gray')
```




    <matplotlib.image.AxesImage at 0x1e014cf5c88>




![png](images/DCToutput_6_1.png)


A naive implementation of the 2-D DCT requires four for loops and has a run time complexity of <p align="center"><img alt="$$O(N^4)$$" src="svgs/8e59a38a22630aad0fdf0b111685fa85.svg" align="middle" width="58.370044pt" height="22.196828pt"/></p> and this can be computationally very expensive when the size of the images scale upwards. Ideally, implementing an algorithm with a much better time complexity is preferred to effectively compute the 2-D DCT in a much shorter time. Below, I have performed the naive implementation of the 2-D DCT to see the different basis functions of the discrete cosine transform.


```python
# Normalization coefficients for DCT
def C(l, R):
    if l == 0:
        return math.sqrt(1.0/R)
    else:
        return math.sqrt(2.0/R)

# Discrete cosine transform
def DCT(p, q, m, n, M, N):
    a = (np.pi*(2*m + 1)*p)/(2*M)
    b = (np.pi*(2*n + 1)*q)/(2*N)
    return C(p, M)*C(q, N)*np.cos(a)*np.cos(b)
```


```python
array = np.zeros((16, 16))
arrayList = []
for x in range(array.shape[1]):
    for y in range(array.shape[0]):
        temporary = np.zeros((array.shape[0], array.shape[1]))
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                temporary[i,j] = DCT(y, x, j, i, array.shape[1], array.shape[0])
        arrayList.append(temporary)
```


```python
fig = plt.figure()
for v in range(len(arrayList)):
    fig.add_subplot(array.shape[0], array.shape[1], 1 + v)
    plt.imshow(arrayList[v], cmap='gray')
    plt.axis('off')
plt.show()
```


![png](images/DCToutput_10_0.png)


## Optimization of the Discrete Cosine Transform

The naive implementation is too computationally expensive and impractical to perform on large images. For example, if I had a 512x512 image, the naive implementation would execute approximately 68,719,476,736 operations to complete the DCT. 

### How can we optimize the DCT?

The answer to the question lies in two different approaches. One route is exploit the symmetry of the DCT and build a fast cosine transform. The other route is to implement the DCT through the Fast Fourier Transform through the radix-2 decimation in time method. By altering the input, one can simply use a FFT algorithm and extract the real coefficients to mimic the output of the DCT.
I decided to choose the latter option because there were more resources and information on FFT algorithms versus DCT algorithms. Additionally, the FFT algorithm is more versatile than the DCT algorithm because later exercises of this project require applying a convolutional kernel on image through Fourier Transforms.

### Fast Fourier Transform Algorithm

The FFT is an algorithm that computes the discrete Fourier Transform in <p align="center"><img alt="$$\mathcal{O}[N\log N]$$" src="svgs/8dc30e429ec0959a7a5e0dfd4c359bd3.svg" align="middle" width="96.227944pt" height="19.92528pt"/></p> versus a naive implementation that runs in <p align="center"><img alt="$$\mathcal{O}[N^2]$$" src="svgs/635dce54d0d10af5f28401f9537eb4b9.svg" align="middle" width="54.606444pt" height="22.196828pt"/></p>.

#### Discrete Fourier Transform
<p align="center"><img alt="$$ X_k = \sum_{n=0}^{N-1} x_n \cdot e^{-i~2\pi~k~n~/~N} $$" src="svgs/f1dd5332a00ecdecde8446a5ea45f672.svg" align="middle" width="246.610616pt" height="57.706026pt"/></p>


#### Inverse Discrete Fourier Transform
<p align="center"><img alt="$$ x_n = \frac{1}{N}\sum_{k=0}^{N-1} X_k e^{i~2\pi~k~n~/~N} $$" src="svgs/6764f799c1b8d3c8a53b5bfe9acb39ec.svg" align="middle" width="246.051588pt" height="58.403394pt"/></p>


The FFT derives its fast time complexity by exploiting the symmetries apparent in the DFT. The algorithm described below to implement the FFT is called the radix-2 decimation-in-time (DIT).

First, we can observe that the DFT can be split into two different summations. 

<p align="center"><img alt="$$&#10;\begin{align}&#10;X_k &amp;= \sum_{n=0}^{N-1} x_n \cdot e^{-i~2\pi~k~n~/~N} \\&#10;    &amp;= \sum_{m=0}^{N/2 - 1} x_{2m} \cdot e^{-i~2\pi~k~(2m)~/~N} + \sum_{m=0}^{N/2 - 1} x_{2m + 1} \cdot e^{-i~2\pi~k~(2m + 1)~/~N} \\&#10;    &amp;= \sum_{m=0}^{N/2 - 1} x_{2m} \cdot e^{-i~2\pi~k~m~/~(N/2)} + e^{-i~2\pi~k~/~N} \sum_{m=0}^{N/2 - 1} x_{2m + 1} \cdot e^{-i~2\pi~k~m~/~(N/2)}&#10;\end{align}&#10;$$" src="svgs/142e377deb8ccb19b9423aa7aa919b69.svg" align="middle" width="710.195088pt" height="202.319732pt"/></p>


Observing the second step, the summation on the left is concerned with even numbered inputs whereas the the summation on the right is concerned with odd numbered inputs. To account for even and odd numbered inputs, the expressions <p align="center"><img alt="$$2m$$" src="svgs/70432f1cdc338828692756fae17d925e.svg" align="middle" width="27.457346pt" height="12.840736pt"/></p> and <p align="center"><img alt="$$2m+1$$" src="svgs/900d1aa1405d9b87366a67f53e886bbe.svg" align="middle" width="61.772982pt" height="14.50119pt"/></p> are used respectively. 

The third step is crucial to the radix-2 DIT algorithm. Focusing on the summation for even-numbered inputs, one can observe that the <p align="center"><img alt="$$2$$" src="svgs/fa950582c899ab6c9aa5a2411f073e56.svg" align="middle" width="9.962678pt" height="12.840736pt"/></p> in <p align="center"><img alt="$$2m$$" src="svgs/70432f1cdc338828692756fae17d925e.svg" align="middle" width="27.457346pt" height="12.840736pt"/></p> can be rewritten into the <p align="center"><img alt="$$N$$" src="svgs/d546e0ee5a0604ca911ec7acb79e5a97.svg" align="middle" width="18.181806pt" height="13.61559pt"/></p> as <p align="center"><img alt="$$N/2$$" src="svgs/58ba38ad25033eb4191ce7613d2d7aee.svg" align="middle" width="36.446676pt" height="19.92528pt"/></p>. This equivalent to the even-numbered summation in the second step and now the even-numbered summation in the third step is now the DFT of only half the input. 

Looking at the odd-numbered summation, we can factor out a <p align="center"><img alt="$$e^{-i~2\pi~k~n~/~N}$$" src="svgs/009e0ac8a697b9c22e0b60b885005045.svg" align="middle" width="119.275384pt" height="18.687764pt"/></p> out of the summation because it has no dependency on m. We can perform the same mathematical trick for the even-numbered summation to the odd-numbered summation and now we get a DFT for the odd-numbered inputs. 

#### Why is segregating the DFT into even and odd-numbered inputs powerful?

After the third step, one can see that the original DFT has been halved into two smaller N/2 DFTs. This can be repeated by segregating the even-indexed inputs from the odd-numbered inputs until the halved DFTs has only one number. The DFT of one input and is simply the input itself. Assuming that the input array to the FFT is <p align="center"><img alt="$$2^n$$" src="svgs/5066e128c87d276fa3b5a18fa95d2e05.svg" align="middle" width="19.812404pt" height="14.232246pt"/></p>, one can observe that the number of stages of dividing the DFT occurs <p align="center"><img alt="$$log(2^N)$$" src="svgs/3ef01afce3f249fbe5a666d42884c1b3.svg" align="middle" width="66.78721pt" height="22.739252pt"/></p> (assuming log base 2) or N times.

After dividing the DFT N times, now the divisions have to rejoin each other to resemble the DFT of the original signal. After the third step from the above expressions, one can notice that the original DFT can be summarized through <img alt="$E_k$" src="svgs/05bf2b668cae2b5ab9a4c170036b069b.svg" align="middle" width="23.51605999999999pt" height="40.846770000000035pt"/> (even-numbered DFT) and <img alt="$O_k$" src="svgs/d558846fb5620bb03a65a3cc2524dfed.svg" align="middle" width="24.005841999999987pt" height="40.846770000000035pt"/> (odd-numbered DFT):
<p align="center"><img alt="$$ X_k = E_k + e^{-\frac{2\pi i}{N}k} O_k $$" src="svgs/f4c461b03b76f9f13764601bb3b5ffea.svg" align="middle" width="182.79962pt" height="23.210086pt"/></p>
where
<p align="center"><img alt="$$ E_k = \sum_{m=0}^{N/2 - 1} x_{2m} \cdot e^{-i~2\pi~k~m~/~(N/2)} $$" src="svgs/6d08ae86b3e244ff35ee32e8a28f2874.svg" align="middle" width="303.425534pt" height="60.351684pt"/></p> and
<p align="center"><img alt="$$ O_k = \sum_{m=0}^{N/2 - 1} x_{2m + 1} \cdot e^{-i~2\pi~k~m~/~(N/2)} $$" src="svgs/3ac19f1ae22322dbb683c4e08f9c31fd.svg" align="middle" width="324.089754pt" height="60.351684pt"/></p>

Remember, the DFT can only calculate the spectral representation of a 1-d signal for frequencies from 0 to the length of the signal. In the case above, <p align="center"><img alt="$$X_k$$" src="svgs/7624e0a73dbed7225b01557dc2a5314b.svg" align="middle" width="25.314886pt" height="16.604352pt"/></p> can technically only be calculate for value of k from <p align="center"><img alt="$$0 \leq k &lt; N/2$$" src="svgs/324149020872ecb9cc7239b6e34a5450.svg" align="middle" width="110.543444pt" height="19.92528pt"/></p>. 

How do we calculate for values of k from N/2 to N?

We can exploit the symmetry of the DFT. One can notice that <p align="center"><img alt="$$X_{N+k}$$" src="svgs/6d36a8486adca625488aa0482678ca4a.svg" align="middle" width="51.663386pt" height="18.264812pt"/></p> is equivalent to <p align="center"><img alt="$$X_N$$" src="svgs/105c88dcac90d047a28d67bd03a56a45.svg" align="middle" width="30.624136pt" height="16.604352pt"/></p>:
<p align="center"><img alt="$$ &#10;\begin{align*}&#10;X_{N + k} &amp;=  \sum_{n=0}^{N-1} x_n \cdot e^{-i~2\pi~(N + k)~n~/~N}\\&#10;          &amp;= \sum_{n=0}^{N-1} x_n \cdot e^{- i~2\pi~n} \cdot e^{-i~2\pi~k~n~/~N}\\&#10;          &amp;= \sum_{n=0}^{N-1} x_n \cdot e^{-i~2\pi~k~n~/~N}&#10;\end{align*}&#10;$$" src="svgs/17ad18794520ec6726e0f017f5fa3fe8.svg" align="middle" width="356.607958pt" height="197.028416pt"/></p>

With this property, we can confidently state the following due to the periodicity of the DFT:
<p align="center"><img alt="$$ E_{k + N/2} = E_k $$" src="svgs/02a6d593b155cf4f9c117042ee320d34.svg" align="middle" width="115.76315pt" height="20.700168pt"/></p>
<p align="center"><img alt="$$ O_{k + N/2} = O_k $$" src="svgs/732b31382a17f7e5be1b6594ee2e1470.svg" align="middle" width="116.742714pt" height="20.700168pt"/></p>

Now, a factor remains outside of the <p align="center"><img alt="$$O_k$$" src="svgs/2f3eaf00cb8921a70c7c449b6630e2bc.svg" align="middle" width="24.005842pt" height="16.604352pt"/></p>. When values of k higher than N/2 are plugged into the <img alt="$O_k$" src="svgs/d558846fb5620bb03a65a3cc2524dfed.svg" align="middle" width="24.005841999999987pt" height="40.846770000000035pt"/>, the following factor transforms accordingly:
<p align="center"><img alt="$$ \begin{matrix} e^{\frac{-2\pi i}{N} (k + N/2)} &amp; = &amp; e^{\frac{-2\pi i k}{N} - {\pi i}} \\&#10;&amp; = &amp; e^{-\pi i} e^{\frac{-2\pi i k}{N}} \\&#10;&amp; = &amp; -e^{\frac{-2\pi i k}{N}}&#10;\end{matrix}$$" src="svgs/6479e1e52b2f24a0ab43ac8e819acff6.svg" align="middle" width="253.452306pt" height="76.94515pt"/></p>

Ultimately, one can rewrite the DFT for <p align="center"><img alt="$$X_k$$" src="svgs/7624e0a73dbed7225b01557dc2a5314b.svg" align="middle" width="25.314886pt" height="16.604352pt"/></p> for k values less than N/2 as
<p align="center"><img alt="$$ X_k =  E_k + e^{-\frac{2\pi i}{N}k} O_k$$" src="svgs/9aca1f0eb17003e7d76209d90fa0b592.svg" align="middle" width="182.79962pt" height="23.210086pt"/></p>

and the DFT for <p align="center"><img alt="$$X_k$$" src="svgs/7624e0a73dbed7225b01557dc2a5314b.svg" align="middle" width="25.314886pt" height="16.604352pt"/></p> for k values greater than N/2 and less than N as 
<p align="center"><img alt="$$ X_{k+\frac{N}{2}} = E_k - e^{-\frac{2\pi i}{N}{k}} O_k$$" src="svgs/4786bfaf4cfe8d830131ca4637fa95c8.svg" align="middle" width="211.616696pt" height="30.32154pt"/></p>

With the following derivations, once can observe that this pattern will continue as long as the DFT's are divided into 2. 


```python
img3 = cv2.imread('2-point DFT.png', 0)
img4 = cv2.imread('4-point DFT.png', 0)
img5 = cv2.imread('8-point DFT.png', 0)
```

### How to recombine the divisions of the DFT to obtain the original DFT?

For example, let's say we are performing the DFT on a 2-sample signal. There is only one even-indexed input and one odd-indexed input. Using the FFT radix-2 DIT algorithm, two separate DFTs can be created where each of these DFTs only contain one input. One of these halved DFTs will contain the even-index input and the other will have odd-index input. Let's call the even-indexed DFT <img alt="$Even_k$" src="svgs/3bc08d03f70ba7bedb9c7f046fe9408d.svg" align="middle" width="56.275227999999984pt" height="40.846770000000035pt"/> and the odd-indexed DFT <img alt="$Odd_k$" src="svgs/fc1e622131d66b529ac6b79921188895.svg" align="middle" width="45.30106799999999pt" height="41.51101199999998pt"/>.

The original 2-point DFT has two values for k for which the DFT will calculate the frequency representation of the signal. Those values are simply <p align="center"><img alt="$$X_0$$" src="svgs/941107b378c585544460c087fe33473a.svg" align="middle" width="24.450058pt" height="16.604352pt"/></p> and <p align="center"><img alt="$$X_1$$" src="svgs/7e1ccff43e4f502a27a0d5ea5e6b1d5c.svg" align="middle" width="24.450058pt" height="16.604352pt"/></p>. When we recombine the single-point DFTs we observe that the original two-point DFT is comprised of 
<p align="center"><img alt="$$ X_0 = Even_0 + e^{-\frac{2\pi i}{N}\cdot0}Odd_0$$" src="svgs/79e920d622461180205a57a3cdb2262c.svg" align="middle" width="238.126954pt" height="23.210086pt"/></p>
and 
<p align="center"><img alt="$$ X_1 = Even_0 - e^{-\frac{2\pi i}{N}\cdot0}Odd_0$$" src="svgs/d467230f2b64173e0d1c48ebbffe7b3f.svg" align="middle" width="238.126954pt" height="23.210086pt"/></p>

These formulas can be easily obtained from the latter two derivations in the previous section (representing <img alt="$X_k$" src="svgs/1a35cf75b6c416e1e4a2b594e79040e6.svg" align="middle" width="25.314885999999987pt" height="40.846770000000035pt"/> and <img alt="$X_{k+N/2}$" src="svgs/b44e99b815e435ad109119efb4f921aa.svg" align="middle" width="66.48287799999999pt" height="40.846770000000035pt"/> through even and odd-numbered DFTs using symmetry).

To calculate a two-point DFT, one simply needs to add the inputs together for the k = 0 and subtract the inputs from each other (not forgetting the factor on the odd-numbered DFT) for k = 1. The DFT for a single point is trivial as it is the single point itself.

The representation for a 2-point DFT can be better represented through a butterfly diagram shown below (alpha = 1):


```python
plt.imshow(img3, cmap='gray')
```




    <matplotlib.image.AxesImage at 0x1e01510d160>




![png](images/DCToutput_14_1.png)


What is the value of alpha?

The value of alpha comes from evaluating the factor attached to the odd numbered DFT. The value of k in that factor depends on the <img alt="$X_k$" src="svgs/1a35cf75b6c416e1e4a2b594e79040e6.svg" align="middle" width="25.314885999999987pt" height="40.846770000000035pt"/> being evaluated. 

#### How is this process scaled for higher numbered-DFT recombinations?

The butterfly diagram for a 4-point and 8-point DFT are shown below.


```python
plt.imshow(img4, cmap='gray')
```




    <matplotlib.image.AxesImage at 0x1e00aac7d68>




![png](images/DCToutput_16_1.png)



```python
plt.figure(figsize=(8,8))
plt.imshow(img5, cmap='gray')
```




    <matplotlib.image.AxesImage at 0x1e00cdafbe0>




![png](images/DCToutput_17_1.png)


#### Bit Reversal

Observe the order of the index of the values in the 4 and 8-point DFT. For the 8-point DFT, the indexes of the input values are 0, 4, 2, 6, 1, 5, 3, and 7. The special ordering of the input values is obtained through bit-reversal. For example, if we had a 8-point DFT, one would iterate through from 0-7 in order (indexes of the values), and for each number, reverse the bits of the index to find the order of the input array that is used by the FFT algorithm. A preprocessing step of the FFT algorithm is to preprocess the input signal such that it follows a bit-reversed order. Once the signal has been processed, one can simply perform the FFT that is similar to that of mergesort. 

### How do we obtain the DCT?

I read "A fast cosine transform in one and two dimensions" by J. Makhoul and understood how to obtain a fast cosine transformation by evolving the FFT. 

The DCT differs from the DFT in the sense that the boundary conditions implied by the DCT are different and only cosines are used in calculation of the spectral representation. So, by altering the symmetry of the input data going into the FFT, one can get an output that mirrors the results of a DCT applied on the original input data.

Makhoul stated that to obtain the DCT of an input signal from an FFT algorithm, pad the original input signal with with N zeros appended at the end (N is the assumed length of the original input signal). 

For example, [1, 2, 3, 4] becomes [1, 2, 3, 4, 0, 0, 0, 0].

Now, take the FFT of that signal, multiply every input of the FFT with <img alt="$2e^{-i\pi k/2N}$" src="svgs/59f9602c994e9e76497f21426eb6e050.svg" align="middle" width="86.17958599999999pt" height="53.074500000000015pt"/> where k is the index of a data point in the signal. Then, retain only the first N values of the FFT, and keep only the real parts of the FFT. This is equivalent to taking the DCT of the original input signal.

Below, I implemented the FFT, butterfly computation, and the DCT from fundamental principles. At the end of the section, I prove my results by comparing it to the dct call from scipy.fftpack.


```python
# Bit reversal of the input data
def bit_reversal(a):
    array = np.zeros(a.shape)
    array = array.astype('complex')
    log = int(math.log(array.size, 2))
    for n in range(a.size):
        b = '{:0{width}b}'.format(n, width=log)
        array[n] = a[int(b[::-1], 2)]
    return array
a = np.array([0,1,2,3,4,5,6,7])
bit_reversal(a)
```




    array([ 0.+0.j,  4.+0.j,  2.+0.j,  6.+0.j,  1.+0.j,  5.+0.j,  3.+0.j,
            7.+0.j])




```python
def ButterflyComputationFFT(input_array):
    output = np.zeros(input_array.shape)
    output = output.astype('complex')
    if input_array.size == 1:
        return input_array[0]
    else:
        N = input_array.size
        twiddle = complex(np.cos(2.0*np.pi/N), -1.0*np.sin(2.0*np.pi/N))
        for i in range(int(N/2)):
            factor = twiddle**(i)
            print("Twiddle Factor: ")
            print(factor)
            output[int(i)] = input_array[int(i)] + factor*input_array[int(i+N/2)]
            output[int(i+N/2)] = input_array[int(i)] + -1.0*factor*input_array[int(i+N/2)]
    return output

def ButterflyComputationFFT_Exp(input_array):
    N = input_array.size
    twiddle = complex(np.cos(2.0*np.pi/N), -1.0*np.sin(2.0*np.pi/N))
    for i in range(int(N/2)):
        factor = twiddle**i
        a = input_array[int(i)]
        input_array[int(i)] = a + factor*input_array[int(i+N/2)]
        input_array[int(i+N/2)] = a + -1.0*factor*input_array[int(i+N/2)]
    return input_array

def ButterflyComputationIFT(input_array):
    output = np.zeros(input_array.shape)
    output = output.astype('complex')
    if input_array.size == 1:
        return input_array[0]
    else:
        N = input_array.size
        half = N/2
        fl = N
        twiddle = complex(np.cos(2.0*np.pi/fl), np.sin(2.0*np.pi/fl))
        for i in range(int(half)):
            factor = twiddle**(i)
            print("Twiddle Factor: ")
            print(factor)
            output[int(i)] = input_array[int(i)] + factor*input_array[int(i+half)]
            output[int(i+half)] = input_array[int(i)] + -1.0*factor*input_array[int(i+half)]
    return output


def ButterflyComputationIFT_DCT(input_array):
    output = np.zeros(input_array.shape)
    output = output.astype('complex')
    if input_array.size == 1:
        return input_array[0]
    else:
        N = input_array.size
        half = N/2
        fl = N
        twiddle = complex(np.cos(2.0*np.pi/fl), np.sin(2.0*np.pi/fl))
        for i in range(int(half)):
            factor = twiddle**(i)
            output[int(i)] = input_array[int(i)] + -1.0*factor*input_array[int(i+half)]
            output[int(i+half)] = input_array[int(i)] + 1.0*factor*input_array[int(i+half)]
    return output
```


```python
# Input array must be a power of 2
def FFT(ar):
    complex_array = np.zeros(ar.shape)
    complex_array = complex_array.astype('complex')
    complex_array = bit_reversal(ar)
    iterations = math.log(ar.size, 2)-1
    while (iterations > -1):
        arrayList = np.split(complex_array, 2**(iterations))
        length = complex_array.size/len(arrayList)
        for j in range(int(len(arrayList))):
            complex_array[int(j*length):int((j+1)*length)] = ButterflyComputationFFT(arrayList[int(j)])
        iterations = iterations - 1
        print("Resultant Array")
        print(complex_array)
    return complex_array

def FFT_Exp(ar):
    complex_array = np.zeros(ar.shape)
    complex_array = complex_array.astype('complex')
    complex_array = bit_reversal(ar)
    iterations = math.log(ar.size, 2)-1
    while (iterations > -1):
        step = complex_array.size/(2**iterations)
        j = 0
        while (j < complex_array.size):
            complex_array[int(j):int(j+step)] = ButterflyComputationFFT_Exp(complex_array[int(j):int(j+step)])
            j = j + step
        iterations = iterations - 1
    return complex_array

# Input array must be a power of 2
def IFFT(ar):
    complex_array = np.zeros(ar.shape)
    complex_array = complex_array.astype('complex')
    complex2_array = bit_reversal(ar)
    iterations = math.log(ar.size, 2)-1
    while (iterations > -1):
        arrayList = np.split(complex2_array, 2**(iterations))
        length = complex2_array.size/len(arrayList)
        for j in range(int(len(arrayList))):
            complex_array[int(j*length):int((j+1)*length)] = ButterflyComputationIFT(arrayList[int(j)])
        iterations = iterations - 1
        complex2_array = complex_array
        print("Resultant Array")
        print(complex2_array)
    return complex_array/complex_array.size

# Input array must be a power of 2
def IFFT_DCT(ar):
    complex_array = np.zeros(ar.shape)
    complex_array = complex_array.astype('complex')
    complex2_array = bit_reversal(ar)
    iterations = math.log(ar.size, 2)-1
    while (iterations > -1):
        arrayList = np.split(complex2_array, 2**(iterations))
        length = complex2_array.size/len(arrayList)
        for j in range(int(len(arrayList))):
            complex_array[int(j*length):int((j+1)*length)] = ButterflyComputationIFT_DCT(arrayList[int(j)])
        iterations = iterations - 1
        complex2_array = complex_array
    return complex_array
```


```python
def ConvertToDCT(fft_input):
    fft = np.zeros((4*fft_input.size))
    fft[1:2*fft_input.size:2] = fft_input
    fft[2*fft_input.size+1::2] = fft_input[::-1]
    dct = np.zeros(fft.shape)
    dct = dct.astype('complex')
    dct = FFT_Exp(fft)
    return dct.real[0:fft_input.size]

def ConvertToDCT_Exp(fft_input):
    fft = np.zeros((2*fft_input.size))
    fft[:fft_input.size] = fft_input
    fft[fft_input.size:] = 0
    dct = np.zeros(fft.shape)
    dct = dct.astype('complex')
    dct = FFT_Exp(fft)
    for i in range(len(dct)):
        dct[i] = 2*dct[i]*np.exp(-1j*np.pi*i/(2*fft_input.size))
    return dct.real[0:fft_input.size]

def ConvertToIDCT(fft_i):
    fft = np.zeros((2*fft_i.size))
    fft[:fft_i.size] = 0
    fft[fft_i.size:] = fft_i
    idct = np.zeros(fft.shape)
    idct = idct.astype('complex')
    idct = IFFT_DCT(fft)
    for i in range(len(idct)):
        idct[i] = 2*idct[i]*np.exp(-1j*np.pi*i/(2*fft_i.size))
    return idct.real[0:fft_i.size]

def DCT_Normalized(dct):
    for i in range(len(dct)):
        if i == 0:
            dct[i] = dct[i]*np.sqrt(1/(4*dct.size))
        else:
            dct[i] = dct[i]*np.sqrt(1/(2*dct.size))
    return dct
```


```python
# Generate 1024 random values
np.random.seed(1)
g = np.random.random(1024)
```


```python
# Check to see if dct and idct from scipy.fftpack returns back
# the original array of 1024 values
d = scipy.fftpack.dct(g, norm = 'ortho')
np.allclose(g, scipy.fftpack.idct(d, norm = 'ortho'))
```




    True




```python
# Check to see if DCT implementation matches up with 
# scipy.fftpack.dct's implementation
np.allclose(ConvertToDCT_Exp(g), scipy.fftpack.dct(g))
```




    True




```python
# Compare the FFT implementation with np.fft.fft's implementation
np.allclose(FFT_Exp(g), np.fft.fft(g))
```




    True




```python
# Compare the IFFT implementation with np.fft.ifft's implementation
f = FFT_Exp(g)
np.allclose(IFFT(f), np.fft.ifft(f))
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-11-a543b6c799a8> in <module>()
          1 # Compare the IFFT implementation with np.fft.ifft's implementation
    ----> 2 f = FFT_Exp(g)
          3 np.allclose(IFFT(f), np.fft.ifft(f))
    

    NameError: name 'FFT_Exp' is not defined

