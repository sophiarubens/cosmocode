import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import ifftshift,fftshift,fft2,ifft2
from scipy.signal import convolve,convolve2d

##
def fft_convolution(image, kernel):
 
    image_padding = len(kernel)//2
    imgpadding=[image_padding,image.shape[0]+image_padding,image_padding,image.shape[1]+image_padding]
    print("image padding indices=",imgpadding)

  #padding image
#   image_pad = np.pad(image_pad,(((image_padding+1)//2,image_padding//2),((image_padding+1)//2, image_padding//2)), mode='edge') 
    image=ifftshift(image)
    image_pad = np.pad(image, image_padding, mode='constant')

  #pad kernel so it is same size as image
    pad_0_low = image_pad.shape[0] // 2 - kernel.shape[0] // 2
    pad_0_high = image_pad.shape[0] - kernel.shape[0] - pad_0_low
    pad_1_low = image_pad.shape[1] // 2 - kernel.shape[1] // 2
    pad_1_high = image_pad.shape[1] - kernel.shape[1] - pad_1_low
    kernel_padding=[pad_0_low,kernel.shape[0]+pad_0_high,pad_1_low,kernel.shape[1]+pad_1_high]
    print("kernel padding indices=",kernel_padding)
    kernel=ifftshift(kernel)
    kernel = np.pad(kernel, ((pad_0_low, pad_0_high),(pad_1_low, pad_1_high)), 'constant')
#   kernel = np.pad(kernel, (((pad_x+1)//2,pad_x//2),((pad_y+1)//2,pad_y//2)), 'constant')


#   #convert both to fft
#     img_fft = fft2(image_pad)
#     kernel_fft = fft2(kernel)

#   #multiply 2 fourier matrices
#     img = img_fft * kernel_fft

#   #inverse fft
#     uncropped = ifft2(img).real
    uncropped=convolve(image_pad,kernel)

    ##
    fig,axs=plt.subplots(1,3,figsize=(15,5))
    for i in range(3):
        axs[i].axhline(kernel_padding[0],c="C0")
        axs[i].axhline(kernel_padding[1],c="C0")
        axs[i].axvline(kernel_padding[2],c="C0")
        axs[i].axvline(kernel_padding[3],c="C0",label="kernel padding bounds")

        axs[i].axhline(imgpadding[0],c="C1",ls=":")
        axs[i].axhline(imgpadding[1],c="C1",ls=":")
        axs[i].axvline(imgpadding[2],c="C1",ls=":")
        axs[i].axvline(imgpadding[3],c="C1",ls=":",label="image padding bounds")

        axs[i].legend()
    im=axs[0].pcolor(image_pad)
    plt.colorbar(im,ax=axs[0])
    axs[0].set_title("padded image")

    im=axs[1].pcolor(kernel)
    plt.colorbar(im,ax=axs[1])
    axs[1].set_title("padded kernel")

    im=axs[2].pcolor(uncropped)
    plt.colorbar(im,ax=axs[2])
    axs[2].set_title("uncropped convolution")
    plt.tight_layout()
    plt.show()
  ##

  #slice array to get rid of padding and original size of image
#   return output[image_padding:len(image_pad),image_padding:len(image_pad)]
    return uncropped,kernel_padding,image_padding
#   return uncropped[image_padding:-image_padding, image_padding:-image_padding]
##

Wcont=np.load("cyl_Wcont.npy")
P=np.load("cyl_P.npy")

Pcont,kernelpadding,imgpadding=fft_convolution(P,Wcont) #image,kernel
imgpadding=(imgpadding,imgpadding+P.shape[0],imgpadding,imgpadding+P.shape[1])

print("Wcont.shape=",Wcont.shape)
print("P.shape=",P.shape)
print("Pcont.shape=",Pcont.shape)

# fig,axs=plt.subplots(1,3,figsize=(15,5))
# for i in range(4):
#    for j in range(3):
#     axs[j].axhline(kernelpadding[i],c="C0")
#     axs[j].axhline(kernelpadding[i],c="C0")
#     axs[j].axvline(kernelpadding[i],c="C0")
#     axs[j].axvline(kernelpadding[i],c="C0",label="kernel padding bounds")

#     axs[j].axhline(imgpadding[i],c="C1")
#     axs[j].axhline(imgpadding[i],c="C1")
#     axs[j].axvline(imgpadding[i],c="C1")
#     axs[j].axvline(imgpadding[i],c="C1",label="image padding bounds")

#     axs[j].legend()

# im=axs[0].pcolor(P)
# plt.colorbar(im,ax=axs[0])
# axs[0].set_title("P pristine")
# im=axs[1].pcolor(Wcont)
# plt.colorbar(im,ax=axs[1])
# axs[1].set_title("Wcont")
# im=axs[2].pcolor(Pcont)
# plt.colorbar(im,ax=axs[2])
# axs[2].set_title("Pcont (actually Ptrue atm)")

# for i in range(2):

#     axs[i].set_xlabel("k$_{||}$ (Mpc$^{-1}$)")
#     axs[i].set_ylabel("k$_\perp$ (Mpc$^{-1}$)")

# plt.suptitle("inspect where the analytical shape issue enters")
# plt.tight_layout()
# plt.savefig("test_with_convolve2d.png")
# plt.show()