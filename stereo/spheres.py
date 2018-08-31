import matplotlib
import numpy as np
from numpy.fft import fftn, ifftn, fftshift, ifftshift
from numpy.linalg import norm
from numpy import abs, isnan, rot90
from matplotlib import pyplot as plt
from skimage import transform
from scipy.misc import imread
from scipy import ndimage
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import os.path
from multiprocessing import Pool
script_dir = os.path.dirname(os.path.abspath(__file__))

#dilation and erosion functions to blur edges of the mask for auto-correlation

def import_image(image,convert=False):
    i = Image.open(os.path.join(script_dir, image))
    if convert == True:
        i = i.convert('L')       #convert to grayscale
    a = np.asarray(i).copy()
    return a

def log(x):
    x[x >= 1] = np.log(x[x >= 1])
    return x

def rebin(a, new_shape):
    """Rebins to an new_shape=(m,n) matrix"""
    M, N = a.shape
    m, n = new_shape
    return a.reshape((m,int(M/m),n,int(N/n))).mean(3).mean(1)

def remap_parallel(distance):
    """Remaps a 45 degree tilted detector at a distance from the object"""
    p = 256
    img_input = np.load('centered_image1.npy')
    mask = np.load('centered_mask1.npy')
    sleeve_mask = np.load('sleeve_mask1.npy')
    a,b = img_input.shape
    N = a # Number of pixels
    du = (2048/N) * 13.5e-6 # pixel size (m)
    beamCentre_pixel = np.array([1100, 998]) / 2048 * N
    x,y = img_input.shape
    dq_array = du/(wavelength*distance)
    X_array = 1 / dq_array
    dx_array = X_array / N

    # Make a meshgrid for the spatial frequency variables u_x and u_y
    ux, uy = np.meshgrid(np.arange(-beamCentre_pixel[0], N - beamCentre_pixel[0]), \
    	                 np.arange(-beamCentre_pixel[1], N - beamCentre_pixel[1]))
    ux = ux * du
    uy = uy * du
    ux_vec = ux.flatten()
    uy_vec = uy.flatten()
    uz_vec = np.zeros([np.size(ux)])

    #-----------------
    # Flat detector geometry mapped onto the Ewald sphere
    d_vec_array = np.sqrt(ux_vec**2 + uy_vec**2 + distance**2)

    qx_array = (1/wavelength) * (ux_vec/d_vec_array)
    qy_array = (1/wavelength) * (uy_vec/d_vec_array)
    qz_array = (1/wavelength) * (distance/d_vec_array - 1)

    #-----------------
    # Make a 3 by N matrix containing all the coordinates
    q_sphere_temp = np.vstack((qx_array,qy_array))
    q_sphere_array = np.vstack((q_sphere_temp,qz_array))

    #-----------------
    # Now rotate all the points
    theta = 0
    s = np.sin(theta)
    c = np.cos(theta)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) # rotation about y

    q_rotated_array = q_sphere_array.copy()

    # Throw away z-component

    # Multiply by distance to interaction point.
    qx_mappedtoflat_array = q_rotated_array[0,:] * d_vec_array * wavelength
    qy_mappedtoflat_array = q_rotated_array[1,:] * d_vec_array * wavelength

    #=====================================================================
    # Interpolate

    q_temp = np.vstack((qx_mappedtoflat_array, qy_mappedtoflat_array))
    q_temp = q_temp.T
    q_rotated_mapped = q_temp
    q_sphere_rotated_mappedToFlat_val_vec = img_input.flatten()
    mask_val_vec = mask.flatten()


    qx_min = np.min(qx_mappedtoflat_array)
    qx_max = np.max(qx_mappedtoflat_array)
    qy_min = np.min(qy_mappedtoflat_array)
    qy_max = np.max(qy_mappedtoflat_array)

    qx_new_array,qy_new_array = np.meshgrid(np.linspace(qx_min,qx_max,x), np.linspace(qy_min,qy_max,y))

    print('Interpolating...')
    remapped_image = griddata(q_rotated_mapped, q_sphere_rotated_mappedToFlat_val_vec, (qx_new_array,qy_new_array),method='linear')
    remapped_mask = griddata(q_rotated_mapped,mask_val_vec,(qx_new_array,qy_new_array),method='linear')
    print('Done interpolating')

    remapped_image[isnan(remapped_image) == True] = 0
    remapped_mask[isnan(remapped_mask) == True] = 0
    remapped_mask[remapped_mask < 0.1] = 0
    remapped_mask[remapped_mask > 0.1] = 1

    print('Now centering...')
    difflist = []
    rolls = []
    #no y shift after remapping, only x
    img_control = remapped_image.copy()
    inv_img = remapped_image.copy()
    inv_mask = remapped_mask.copy()
    for j in range(2):
        inv_img = rot90(inv_img)
        inv_mask = rot90(inv_mask)
    for i in range(80):
        img1 = np.roll(inv_img,-(i+1))
        mask1 = np.roll(inv_mask,-(i+1))
        mask1 *= remapped_mask*sleeve_mask
        difflist.append(norm((img1 - remapped_image)*mask1))
        rolls.append(-(i+1))
    best = np.argmin(difflist)
    roll = rolls[best]
    shifted_remapped_image = np.roll(remapped_image,-int(roll/2))
    shifted_remapped_mask = np.roll(remapped_mask,-int(roll/2))
    print('Done centering.')

    if visualize == 1:
        plt.subplot(121)
        plt.imshow(log(shifted_remapped_image),cmap='viridis')
        rgb_image = np.zeros([p,p,3])
        rgb_image[:,:,0] = shifted_remapped_image
        img_rotate = shifted_remapped_image.copy()
        for j in range(2):
            img_rotate = rot90(img_rotate)
        rgb_image[:,:,2] = img_rotate
        rgb_image[:,:,0] *= 1/np.max(rgb_image[:,:,0])
        rgb_image[:,:,2] *= 1/np.max(rgb_image[:,:,2])
        plt.subplot(122)
        plt.imshow(rgb_image)
        plt.show()

    #invert resulting images
    flipped_image = shifted_remapped_image
    flipped_mask = shifted_remapped_mask
    for j in range(2):
        flipped_image = rot90(flipped_image)
        flipped_mask = rot90(flipped_mask)

    #temp is mix of both masks, take difference of resulting image and its inversion
    #to find most centrosymmetric result
    temp = np.ones_like(shifted_remapped_image)
    temp_total = (shifted_remapped_mask + flipped_mask)*sleeve_mask
    temp[temp_total != 2] = 0
    difference = (shifted_remapped_image - flipped_image)*sleeve_mask
    rotate_diff = norm(temp*difference)/norm(temp*shifted_remapped_image)
    #pool_on to test for best distance in parallel, output_packet is both distance
    #and error difference because only a single input and output seem to work with pool.map()
    output_packet = np.array([distance,rotate_diff])
    if pool_on == 1:
        return output_packet
    elif pool_on == 0:
        return shifted_remapped_image,shifted_remapped_mask


def cornermask(mask):
    """Applies masks over left and right corners"""
    x,y = mask.shape
    radius = int(7/18*x)
    for i in range(x):
        for j in range(y):
            if np.sqrt(i**2+j**2) < radius:
                mask[i,j] = 0
            if np.sqrt((i-x)**2+j**2) < radius:
                mask[i,j] = 0
            if np.sqrt(i**2+(j-y)**2) < radius:
                mask[i,j] = 0
            if np.sqrt((i-x)**2+(j-y)**2) < radius:
                mask[i,j] = 0
    return mask

def sleeve_mask(mask):
    """Applies 20-pixel mask over boundaries"""
    x,y = mask.shape
    i = int(20*x/128)
    mask[:i] = 0
    mask[:,:i] = 0
    mask[x-i:] = 0
    mask[:,y-i:] = 0
    return mask

def beamstop(mask):
    """Applies beamstop mask over center"""
    x,y = mask.shape
    radius = int(6/72*x)
    center = int(x/2)
    for i in range(x):
        for j in range(y):
            if np.sqrt((i-center)**2+(j-center)**2) < radius:
                mask[i,j] = 0
    return mask

def mirrormask(mask):
    """Applies splotch masks particular to a certain diffraction sample set"""
    x,y = mask.shape
    for i in range(x):
        for j in range(y):
            if i >= (-55/90*j+120):
                mask[j,i] = 0
    mask[17:32,49:83] = 0
    return mask

def blur(psi,phase_factor):
    psi2 = phase_factor * fftshift(fftn(psi))
    psi2 = ifftn(ifftshift(psi2))
    return psi2

def focus(psi,inverse_phase):
    psi2 = inverse_phase * fftshift(fftn(psi))
    psi2 = ifftn(ifftshift(psi2))
    return psi2

def argue(z):
    k_z = np.sqrt(k**2 - k_x_mesh**2 - k_y_mesh**2)
    arg = z*k_z
    return arg

def phase(arg):
    #propagate fourier space array forward by z length units
    phase_factor = np.exp(1j*arg)
    return phase_factor

#Parameters
test_ps = 0
test_fs = 0
test_pm = 0
test_fm = 0
test_iter = 0

volume_support = 1 #to apply volume support once every so often
full_volume = 0 #to apply volume support with every p_s operation
first_time = 1 #for a newly imported image -- subtracts noise and centers
noise = 0 #removes noise on its own -- not recommended if image is already remapped
noise_median = 0 #removes median of left and right halves
noise_poly_fit = 1 #fits a polynomial to the noise and subtracts it out -- also not recommended
second_order = 0
third_order = 0
fourth_order = 1
fifth_order = 0
sixth_order = 0
no_fringe = 0 #crops out center of intensity pattern
visualize = 0#for debugging
ewald_remap = 0 #to remap a 45 degree tilted detector
pool_on = 1 #set to 1 to do a detector distance sweep with remapping
overwrite = 0 #to overwrite files saved to disk
shrinkwrapped = 0 # for implementing a shrinkwrap support later
simulation = 0 #for simulating a reconstruction with no fringes

n = 256

if first_time == 1:
    img_1 = imread('0008402_20140502_044428.png')
    n,m = img_1.shape
    img_1 = np.roll(img_1,60,axis=0)
    mask = np.ones_like(img_1)
    mask = sleeve_mask(beamstop(mask))
    mask = ndimage.binary_dilation(mask,structure=np.ones([2,2])).astype(np.float32)
    x_mesh,y_mesh = np.meshgrid(np.arange((-n/2),(n/2)),np.arange((-n/2),(n/2)))
    gaussian = np.exp(-(x_mesh**2 + y_mesh**2))
    mask = fftshift(abs(ifftn(fftn(mask)*fftn(gaussian))))
    mask *= 1/np.max(mask)
    plt.subplot(221)
    plt.imshow(np.log10(img_1),cmap='viridis')
    plt.title('Log10 of Sphere Diffraction')
    plt.colorbar()
    plt.subplot(222)
    plt.imshow(mask*np.log10(img_1),cmap='viridis')
    plt.title('Log10 of Sphere Diffraction with Mask')
    plt.colorbar()
    plt.subplot(223)
    plt.imshow(np.log10(abs(fftshift(ifftn(img_1)))),cmap='viridis')
    plt.title('Log10 of Auto-Correlation of Unmasked Intensities')
    plt.colorbar()
    plt.subplot(224)
    plt.imshow(np.log10(abs(fftshift(ifftn(mask*img_1)))),cmap='viridis')
    plt.title('Log10 of Auto-Correlation of Masked Intensities')
    plt.colorbar()
    plt.show()
    q
    #subtract out noise:
    pic1 = imread('pic1.png')
    pic2 = imread('pic2.png')
    pic3 = imread('pic3.png')
    pic4 = imread('pic4.png')
    pic5 = imread('pic5.png')
    pic6 = imread('pic6.png')
    pic7 = imread('pic7.png')
    pic8 = imread('pic8.png')
    pic9 = imread('pic9.png')
    pic10 = imread('pic10.png')
    pic11 = imread('pic11.png')
    noise_image = (pic1 + pic2 + pic3 + pic4 + pic5 + pic6 + pic7 + pic8 + pic9 + pic10 + pic11)/11
    scale_factor = np.median(img_1)/np.median(noise_image)
    img_1 = abs(img_1 - noise_image*scale_factor)
    img_1 = img_1[24:2024,24:2024]
    p,b = img_1.shape
    img_1_noise = img_1.copy()
    threshold = 5e-8
    img_1_noise[img_1 > threshold] = 0
    img_1_noise[970:1185,880:1100] = 0
    img_1_eroded = np.zeros_like(img_1_noise)
    img_1_eroded[img_1_noise == 0] = 1
    img_1_noise = img_1_eroded.copy()
    img_1_eroded = ndimage.binary_erosion(img_1_eroded,structure=np.ones([5,5])).astype(np.float32)
    img_1_dilated = ndimage.binary_dilation(img_1_eroded,structure=np.ones([5,5])).astype(np.float32)
    img_1_masked = img_1*img_1_dilated
    # img_1 = rebin(img_1,(n,n))
    # noise_image = rebin(noise_image,(n,n))

    #import masks:
    blindspot_mask = imread('blindspot_mask.png',mode='L')
    mask_1 = imread('mask1.png',mode='L')
    mask_1 = transform.resize(mask_1,(2048,2048))
    mask_1 = mask_1[24:2024,24:2024]
    blindspot_mask = blindspot_mask[24:2024,24:2024]
    mask_1[mask_1 > 0.1] = 1
    mask_1[mask_1 <= 0.1] = 0
    mask_1[940:1303,850:1150] = 1
    blindspot_mask[blindspot_mask > 0.9] = 1
    blindspot_mask[blindspot_mask < 0.9] = 0

    # mask_1 = transform.resize(mask_1,(p,p))
    # blindspot_mask = transform.resize(blindspot_mask,(p,p))
    mask_1[blindspot_mask == 1] = 0

    if noise_poly_fit == 1:
        print('Now fitting...')
        median_array = np.zeros_like(img_1)
        param = int(len(img_1)/15)
        for i in range(15):
            x_coordinate = int((2*i+1)*param/2)
            for j in range(15):
                y_coordinate = int((2*j+1)*param/2)
                img_temp = img_1_masked[i*param:(i+1)*param,j*param:(j+1)*param]
                median_array[x_coordinate,y_coordinate] = np.median(img_temp[img_temp != 0])
        x = np.linspace(0,(p-1),p)
        y = np.linspace(0,(p-1),p)
        X, Y = np.meshgrid(x, y, copy=False)
        X = X.flatten()
        Y = Y.flatten()
        # noise_array = img_1*img_1_dilated
        # B = noise_array.flatten()
        B = median_array.flatten()
        X = X[B > 0]
        Y = Y[B > 0]
        if second_order == 1:
            A = np.array([X*0+1, X, Y, X**2, X**2*Y, X**2*Y**2, Y**2, X*Y**2, X*Y]).T
        if third_order == 1:
            A = np.array([X*0+1,X,Y,X**2,Y**2,X**3,Y**3,X*Y,X*Y**2,X*Y**3,\
            X**2*Y,X**2*Y**2,X**2*Y**3,X**3*Y,X**3*Y**2,X**3*Y**3]).T
        if fourth_order == 1:
            A = np.array([X*0+1,X,Y,X**2,Y**2,X**3,Y**3,X**4,Y**4,X*Y,X*Y**2,X*Y**3,\
            X*Y**4,X**2*Y,X**2*Y**2,X**2*Y**3,X**2*Y**4,X**3*Y,X**3*Y**2,X**3*Y**3,\
            X**3*Y**4,X**4*Y,X**4*Y**2,X**4*Y**3,X**4*Y**4]).T
        if fifth_order == 1:
            A = np.array([X*0+1,X,Y,X**2,Y**2,X**3,Y**3,X**4,Y**4,X**5,Y**5,\
            X*Y,X*Y**2,X*Y**3,X*Y**4,X*Y**5,X**2*Y,X**2*Y**2,\
            X**2*Y**3,X**2*Y**4,X**2*Y**5,X**3*Y,X**3*Y**2,X**3*Y**3,\
            X**3*Y**4,X**3*Y**5,X**4*Y,X**4*Y**2,X**4*Y**3,X**4*Y**4,\
            X**4*Y**5,X**5*Y,X**5*Y**2,X**5*Y**3,X**5*Y**4,X**5*Y**5,]).T
        if sixth_order == 1:
            A = np.array([X*0+1,X,Y,X**2,Y**2,X**3,Y**3,X**4,Y**4,X**5,Y**5,\
            X**6,Y**6,X*Y,X*Y**2,X*Y**3,X*Y**4,X*Y**5,X*Y**6,X**2*Y,X**2*Y**2,\
            X**2*Y**3,X**2*Y**4,X**2*Y**5,X**2*Y**6,X**3*Y,X**3*Y**2,X**3*Y**3,\
            X**3*Y**4,X**3*Y**5,X**3*Y**6,X**4*Y,X**4*Y**2,X**4*Y**3,X**4*Y**4,\
            X**4*Y**5,X**4*Y**6,X**5*Y,X**5*Y**2,X**5*Y**3,X**5*Y**4,X**5*Y**5,\
            X**5*Y**6,X**6*Y,X**6*Y**2,X**6*Y**3,X**6*Y**4,X**6*Y**5,X**6*Y**6]).T
        print('Fitting com-')
        coeff, r, rank, s = np.linalg.lstsq(A, B[B > 0])
        noisy = np.zeros_like(img_1)
        for i in range(len(noisy)):
            if np.mod(i,1000) == 0:
                print('Fitting com-')
            for j in range(len(noisy.T)):
                if second_order == 1:
                    A = np.array([i*0+1, i, j, i**2, i**2*j, i**2*j**2, j**2, i*j**2, i*j]).T
                if third_order == 1:
                    A = np.array([i*0+1,i,j,i**2,j**2,i**3,j**3,i*j,i*j**2,i*j**3,\
                    i**2*j,i**2*j**2,i**2*j**3,i**3*j,i**3*j**2,i**3*j**3]).T
                if fourth_order == 1:
                    A = np.array([i*0+1,i,j,i**2,j**2,i**3,j**3,i**4,j**4,i*j,i*j**2,i*j**3,\
                    i*j**4,i**2*j,i**2*j**2,i**2*j**3,i**2*j**4,i**3*j,i**3*j**2,i**3*j**3,\
                    i**3*j**4,i**4*j,i**4*j**2,i**4*j**3,i**4*j**4]).T
                if fifth_order == 1:
                    A = np.array([i*0+1,i,j,i**2,j**2,i**3,j**3,i**4,j**4,i**5,j**5,\
                    i*j,i*j**2,i*j**3,i*j**4,i*j**5,i**2*j,i**2*j**2,\
                    i**2*j**3,i**2*j**4,i**2*j**5,i**3*j,i**3*j**2,i**3*j**3,\
                    i**3*j**4,i**3*j**5,i**4*j,i**4*j**2,i**4*j**3,i**4*j**4,\
                    i**4*j**5,i**5*j,i**5*j**2,i**5*j**3,i**5*j**4,i**5*j**5,]).T
                if sixth_order == 1:
                    A = np.array([i*0+1,i,j,i**2,j**2,i**3,j**3,i**4,j**4,i**5,j**5,\
                    i**6,j**6,i*j,i*j**2,i*j**3,i*j**4,i*j**5,i*j**6,i**2*j,i**2*j**2,\
                    i**2*j**3,i**2*j**4,i**2*j**5,i**2*j**6,i**3*j,i**3*j**2,i**3*j**3,\
                    i**3*j**4,i**3*j**5,i**3*j**6,i**4*j,i**4*j**2,i**4*j**3,i**4*j**4,\
                    i**4*j**5,i**4*j**6,i**5*j,i**5*j**2,i**5*j**3,i**5*j**4,i**5*j**5,\
                    i**5*j**6,i**6*j,i**6*j**2,i**6*j**3,i**6*j**4,i**6*j**5,i**6*j**6]).T
                noisy[i,j] = np.dot(A,coeff)
        print('Fitting complete')
        scale_factor = np.median(img_1)/np.median(noisy)
        img_1 = abs(img_1 - noisy*scale_factor)
    # plt.subplot(121)
    # plt.imshow(np.log10(img_1),cmap='viridis')
    # plt.title('Log10 of Real Data Pattern')
    # plt.colorbar()
    # plt.subplot(122)
    # plt.imshow(noisy,cmap='viridis')
    # plt.title('6th Order Polynomial Fit')
    # plt.colorbar()
    # plt.show()
    # q


    #set up zeros around the perimeter:
    sleeve_mask = np.ones([2048,2048])
    sleeve_mask[:,:679] = 0
    sleeve_mask[:,1369:] = 0
    sleeve_mask[:539] = 0
    sleeve_mask[1509:] = 0
    sleeve_mask = transform.resize(sleeve_mask,(p,p))
    img_control = img_1.copy()
    #invert the images and rotate 180 degrees
    inv_img = img_1.copy()
    inv_mask = mask_1.copy()
    for j in range(2):
        inv_img = rot90(inv_img)
        inv_mask = rot90(inv_mask)
    #set up error lists
    difflist = []
    difflisty = []
    rolls = []
    rollsy = []
    for i in range(200):
        img1 = np.roll(inv_img,-(i+1))
        mask1 = np.roll(inv_mask,-(i+1))
        #tempxdifflist is appended with the lowest error for each pixel's x coordinate
        tempxdifflist = []
        for j in range(200):
            img2 = np.roll(img1,(j+1),axis=0)
            mask2 = np.roll(mask1,(j+1),axis=0)
            #mask2 mask is pre-rolled mask, rolled mask, and zero perimeter
            mask2 *= mask_1*sleeve_mask
            #attempts at using pearson correlation
            # c = np.mean(img2)
            # f = (img2 - c)*sleeve_mask
            difflisty.append(norm((img2-img_1)*sleeve_mask))
            # difflisty.append(np.sum(f*g)/np.sqrt(np.sum(f**2)*np.sum(g**2)))
            rollsy.append((j+1))
            tempxdifflist.append(norm((img2 - img_1)*sleeve_mask))
            # tempxdifflist.append(np.sum(f*g)/np.sqrt(np.sum(f**2)*np.sum(g**2)))
        difflist.append(np.min(tempxdifflist))
        rolls.append(-(i+1))
        print(i)
    #pick out best x and y shifts and roll by half that to centralize
    bestx = np.argmin(difflist)
    xroll = rolls[bestx]
    besty = np.argmin(difflisty)
    yroll = rollsy[besty]
    img_1 = np.roll(img_1,-int(xroll/2))
    mask_1 = np.roll(mask_1,-int(xroll/2))
    img_1 = np.roll(img_1,-int(yroll/2),axis=0)
    mask_1 = np.roll(mask_1,-int(yroll/2),axis=0)

    if overwrite == 1:
        np.save('centered_image2000_4poly',img_1)
        np.save('centered_mask2000_4poly',mask_1)
        np.save('sleeve_mask2000_4poly',sleeve_mask)

    plt.figure(1)
    plt.subplot(121)
    plt.plot(rolls,difflist,'o')
    plt.subplot(122)
    plt.plot(rollsy,difflisty,'o')
    plt.figure(2)
    plt.subplot(121)
    plt.imshow(log(img_1)*sleeve_mask,cmap='viridis')
    rgb_image = np.zeros([p,p,3])
    rgb_image[:,:,0] = img_1
    img_rotate = img_1.copy()
    for j in range(2):
        img_rotate = rot90(img_rotate)
    rgb_image[:,:,2] = img_rotate
    rgb_image *= 1/np.max(rgb_image)
    plt.subplot(122)
    plt.imshow(rgb_image)
    plt.show()
    q

if __name__ == '__main__':
    if ewald_remap == 1:
        img_input = np.load('centered_image2000_4poly.npy')
        mask = np.load('centered_mask2000_4poly.npy')
        sleeve_mask = np.load('sleeve_mask2000_4poly.npy')
        p = 256
        img_1 = transform.resize(img_1,(p,p))
        mask = transform.resize(mask,(p,p))
        sleeve_mask = transform.resize(sleeve_mask,(p,p))
        scope = 0.1
        distance = 70e-3
        wavelength = 1.349e-8
        d = 3
        distances = np.linspace((1-scope)*distance,(1+scope)*distance,d)
        a,b = img_input.shape
        N = a # Number of pixels
        du = (2048/N) * 13.5e-6 # pixel size (m)
        pool = Pool(processes=3,maxtasksperchild=1)
        output_array = pool.map(remap_parallel,(distances))
        pool.close()
        pool.join()

        output_array = np.asarray(output_array)
        print(output_array.shape)
        print(output_array)

        distances = output_array[:,0]
        rotate_diff_array = output_array[:,1]

        best = np.argmin(rotate_diff_array)
        best_distance = distances[best]
        print('The best distance is ', best_distance)

        plt.figure(2)
        plt.plot(distances,rotate_diff_array)
        plt.xlabel('Distance in meters')
        plt.ylabel('Error')
        plt.show()
        #since only one output works with pool.map, get the best distance and use it
        #one more time to get the actual remapped image and mask
        pool_on = 0
        best_image,best_mask = remap_parallel(best_distance)
        rgb_image = np.zeros([p,p,3])
        rgb_image[:,:,0] = best_image
        img_rotate = best_image.copy()
        for j in range(2):
            img_rotate = rot90(img_rotate)
        rgb_image[:,:,2] = img_rotate
        rgb_image[:,:,0] *= 1/np.max(rgb_image[:,:,0])
        rgb_image[:,:,2] *= 1/np.max(rgb_image[:,:,2])
        plt.figure(2)
        plt.imshow(rgb_image)
        plt.show()

        if overwrite == 1:
            np.save('remappedoutput1',best_image)
            np.save('remappedmask1',best_mask)

        plt.figure(1)
        plt.subplot(211)
        plt.imshow(np.log(best_image),cmap='gray')
        plt.title('Remapped Image')
        plt.subplot(212)
        plt.imshow(best_mask,cmap='gray')
        plt.title('Remapped Mask')
        plt.show()
        q



cfel = import_image('stereo_cfel1.png',True)
cfel_crop = cfel[30:146,40:156]

d = int(n/2)
cfel_crop = cfel_crop/np.max(cfel_crop)
cfel_crop = transform.resize(cfel_crop,(d,d))

#set up out-of-phase image
cfel_toblur = np.zeros([n,n])
cfel_toblur[:d,:d] = cfel_crop

z = 10e-6
k = 2*np.pi/13.49e-8
k_x_mesh, k_y_mesh = np.meshgrid(np.arange(int(-n/2),int(n/2)),np.arange(int(-n/2),int(n/2)))
#propagate fourier space array forward by z length units
arg = argue(z)
phase_factor = phase(arg)
inverse_phase = np.conj(phase_factor)
cfel_blurred = blur(cfel_toblur,phase_factor)

#set up in-phase image
cfel_focused = np.zeros([n,n])
cfel_focused[d:2*d,d:2*d] = cfel_crop
cfel_focused = transform.resize(cfel_focused,(n,n))
cfel = cfel_blurred.copy()
cfel += cfel_focused
cfel *= 0.5
cfel = abs(cfel)

I_unmapped = imread('good3.png')
I = np.load('remappedoutput1.npy')
I = np.fliplr(rot90(I))
I = np.roll(I,2) #still isn't centered the way I like it so I shift it
I = np.roll(I,2,axis=0)
# I = rebin(I,(n,n))
I = transform.resize(I,(n,n))

I[I < 0] = 1
I[isnan(I) == True] = 1
n = 256

if simulation == 1:
    I_sim = fftshift(abs(fftn(cfel_sim))**2)
    I_sim = rebin(I_sim,(n,n))
    I_sim = I_sim[28:100,28:100]
    I = I[28:100,28:100]
    I_real = I.copy()
    I = I_sim.copy()
    n = 72
    # plt.subplot(121)
    # plt.imshow(rot90(log(I_sim)),cmap='viridis')
    # plt.title('Simulated Intensity')
    # plt.subplot(122)
    # plt.imshow(log(I),cmap='viridis')
    # plt.title('Data')
    # plt.show()
    # q

if no_fringe == 1:
    a,b = I.shape
    I = I[int(a/4):int(3*a/4),int(b/4):int(3*b/4)]
    n *= 0.5
    cfel = transform.resize(cfel,(n,n))

if noise == 1:
    pic1 = imread('pic1.png')
    pic2 = imread('pic2.png')
    pic3 = imread('pic3.png')
    pic4 = imread('pic4.png')
    pic5 = imread('pic5.png')
    pic6 = imread('pic6.png')
    pic7 = imread('pic7.png')
    pic8 = imread('pic8.png')
    pic9 = imread('pic9.png')
    pic10 = imread('pic10.png')
    pic11 = imread('pic11.png')
    noise_image = (pic1 + pic2 + pic3 + pic4 + pic5 + pic6 + pic7 + pic8 + pic9 + pic10 + pic11)/11
    noise_image = rebin(noise_image,(n,n))
    scale_factor = np.median(I)/np.median(noise_image)
    I = abs(I - noise_image*scale_factor)


if noise_median == 1:
    mask = np.ones_like(I)
    mask = cornermask(mask)
    noise_med_left = np.median(I[mask == 0])
    noise_med_right = np.median(I[mask == 2])
    noise_med = 0.5*(noise_med_left+noise_med_right)
    I[:,:36] -= noise_med_left
    I[:,36:] -= noise_med_right
I[I < 0] = 0

if noise_poly_fit == 1:
    x = np.linspace(0,(n-1),n)
    y = np.linspace(0,(n-1),n)
    X, Y = np.meshgrid(x, y, copy=False)

    X = X.flatten()
    Y = Y.flatten()

    A = np.array([X*0+1, X, Y, X**2, X**2*Y, X**2*Y**2, Y**2, X*Y**2, X*Y]).T
    B = noise_image.flatten()

    coeff, r, rank, s = np.linalg.lstsq(A, B)

    noisy = np.zeros_like(I)
    for i in range(len(noisy)):
        for j in range(len(noisy.T)):
            A = np.array([i*0+1, i, j, i**2, i**2*j, i**2*j**2, j**2, i*j**2, i*j])
            noisy[i,j] = np.dot(A,coeff)
    scale_factor = np.median(I)/np.median(noisy)
    # I = np.abs(I - noisy*scale_factor)
    plt.imshow(noisy)
    plt.show()

fringe_mask = np.zeros_like(I)
# if simulation == 0:
#     med = np.median(I)
#     fringe_mask[I > med*7] = 1
if simulation == 1:
    med = np.median(I_real)
    fringe_mask[I_real > med*2] = 1
    fringe_mask = ndimage.binary_erosion(fringe_mask,structure=np.ones([3,3])).astype(np.float32)
    fringe_mask = ndimage.binary_dilation(fringe_mask,structure=np.ones([3,3])).astype(np.float32)
mask_array = np.ones_like(I)
mask_array = beamstop(mask_array)
mask_array[isnan(I) == True] = 0
mask_array2 = ndimage.binary_erosion(mask_array,structure=np.ones([2,2])).astype(np.float32)
x_mesh,y_mesh = np.meshgrid(np.arange((-n/2),(n/2)),np.arange((-n/2),(n/2)))
gaussian = np.exp(-(x_mesh**2 + y_mesh**2))
mask_array3 = abs(ifftn(fftn(mask_array2)*fftn(gaussian)))
mask_array3 *= 1/np.max(mask_array3)
mask_array = mask_array3.copy()
mask_array[mask_array < 1e-10] = 0
mask_array = ifftshift(sleeve_mask(cornermask(fftshift(mask_array))))
# mask_array *= ifftshift(fringe_mask)
mask_array[mask_array > 1-1e-10] = 1
# mask_array *= fringe_mask
I = ifftshift(I)
auto = ifftn(ifftshift(I*mask_array))

if visualize == 1:
    plt.figure(1)
    plt.subplot(241)
    plt.imshow(fftshift(log(I)),cmap='viridis')
    plt.title('Unmasked Intensitites')
    plt.subplot(242)
    plt.imshow(fftshift(mask_array*np.log(I)),cmap='viridis')
    plt.title('Masked Intensities')
    plt.subplot(243)
    plt.imshow(fftshift(np.log(abs(ifftn(fftshift(mask_array*I))))))
    plt.title('Data Auto-correlation')
    plt.subplot(244)
    plt.imshow(cfel)
    plt.title('Non-rotated CFEL')
    plt.subplot(245)
    plt.imshow(fftshift(log(abs(fftn(cfel))**2)))
    plt.title('Simulated Intensities')
    plt.subplot(246)
    plt.imshow(fftshift(log(mask_array*abs(fftn(cfel))**2)))
    plt.title('Masked Simulated Intensities')
    plt.subplot(247)
    plt.imshow(fftshift(np.log(abs(ifftn(mask_array*abs(fftn(cfel))**2)))))
    plt.title('Simulated Auto-correlation')
    plt.subplot(248)
    plt.imshow(cfel)
    plt.title('Rotated by 5 degrees')
    if overwrite == 1:
        plt.savefig('simresults081718.png')
    plt.show()
    q



volume = np.array([0,0])
for i in range(n):
    for j in range(n):
        if cfel_toblur[i,j] != 0:
            volume[0] += 1
for i in range(n):
    for j in range(n):
        if cfel_focused[i,j] != 0:
            volume[1] += 1
print(volume)


wavelength = 13.49e-08
k = 2*np.pi/wavelength
c = 1
beta = 0.9
alphaM = 1/beta
alphaS = -1/beta
s = np.zeros([2,256,256])
s[0,86:120,:62] = 1
s[0,6:16,118:126] = 1
s[1,86+d:120+d,d:62+d] = 1
s[1,6+d:16+d,118+d:126+d] = 1
# s = rot90(s)
s = transform.resize(s,(2,n,n))
# plt.subplot(131)
# plt.imshow(s[0])
# plt.subplot(132)
# plt.imshow(cfel)
# plt.subplot(133)
# plt.imshow(s[1])
# plt.show()
# q

if shrinkwrapped == 1:
    threshold = 0.75

    def shrinkwrap(x):
        # Do the convolution with FFTs
        x_mod = np.abs(ifftn(fftn(x)*Conv_kernel))

        # Make everything larger than a threshold 1 and less than or equal to the threshold zero
        s_new = x_mod > (threshold*np.max(x_mod))

        return s_new

mags = np.sqrt(I)

def volume_density(psi,volume):
    uniques = np.unique(psi[np.real(psi) > 1e-15])
    if uniques.size >= volume:
        threshold = uniques[-volume]
    elif uniques.size < volume:
        threshold = uniques[0]
    x,y = np.where(psi >= threshold)
    return x,y

def p_s(psi):
    """Support Size Constraint. Can Incorporate Volume Support"""
    psi *= s
    # for j in range(2):
    #     if j == 0:
    #         psi[j] = s[j]*psi[j]
    #     elif j == 1:
    #         psi[j] = focus(psi[j],inverse_phase)
    #         psi[j] = s[j]*psi[j]
    #         psi[j] = blur(psi[j],phase_factor)
    psi1 = psi.copy()
    if full_volume == 1:
        psi1 = np.zeros_like(psi)
        for j in range(2):
            if j == 0:
                x,y = volume_density(abs(psi[j]),volume[j])
                for i in range(x.size):
                    psi1[j,x[i],y[i]] = psi[j,x[i],y[i]]
            elif j == 1:
                psi[j] = focus(psi[j],inverse_phase)
                x,y = volume_density(abs(psi[j]),volume[j])
                for i in range(x.size):
                    psi1[j,x[i],y[i]] = psi[j,x[i],y[i]]
                psi1[j] = blur(psi1[j],phase_factor)
                psi[j] = blur(psi[j],phase_factor)
    return psi1

def p_s_vol(psi):
    psi1 = np.zeros_like(psi)
    for j in range(2):
        x,y = volume_density(abs(psi[j]),volume[j])
        for i in range(x.size):
            psi1[j,x[i],y[i]] = psi[j,x[i],y[i]]
    # for j in range(2):
    #     if j == 0:
    #         x,y = volume_density(abs(psi[j]),volume[j])
    #         for i in range(x.size):
    #             psi1[j,x[i],y[i]] = psi[j,x[i],y[i]]
    #     elif j == 1:
    #         psi[j] = focus(psi[j],inverse_phase)
    #         x,y = volume_density(abs(psi[j]),volume[j])
    #         for i in range(x.size):
    #             psi1[j,x[i],y[i]] = psi[j,x[i],y[i]]
    #         psi1[j] = blur(psi1[j],phase_factor)
    #         psi[j] = blur(psi[j],phase_factor)
    return psi1

def f_s(psi):
    """Applies f_s, or R_s, operator"""
    C = p_s(psi)
    D = (1+alphaS)*C - alphaS*psi
    return D

def p_m(psi):
    """Fourier Magnitude Constraint"""
    c1 = np.zeros_like(psi[0])
    D = np.zeros_like(psi)
    for j in range(2):
        c1 += 0.5*(psi[j])
    C1 = fftn(c1)
    C1[mask_array == 1] = mags[mask_array == 1]*np.exp(1j*np.angle(C1[mask_array == 1]))
    B = ifftn(C1)
    for j in range(2):
        D[j] = psi[j]+B-c1
    return D

def f_m(psi):
    """Applies f_m, or R_m, operator"""
    C = p_m(psi)
    D = (1+alphaM)*C - alphaM*psi
    return D

def iterative_cycle(psi, phase_factor=None, inverse_phase=None):
    """Difference Map Iterations"""
    fm_psi = f_m(psi)
    fs_psi = f_s(psi)

    ps_fm_psi = p_s(fm_psi)
    pm_fs_psi = p_m(fs_psi)


    delta_psi = ps_fm_psi - pm_fs_psi

    psi = psi + beta*delta_psi

    return psi

def rms_intensities(psi):
    psi1 = p_s(psi)
    c1 = 0.5*(psi1[0]+psi1[1])
    C1 = fftn(c1)
    I1 = abs(C1)**2
    error = np.sqrt(np.sum((I1[mask_array == 1]-I[mask_array == 1])**2)/np.sum(I[mask_array == 1]**2))
    return error

def cycle_paralleler(psi):
    """Packaged into a function to allow parallelization"""
    i = 5
    j = 1000
    test_error = rms_intensities(psi)
    print(test_error)
    for b in range(i):
        # for k in range(j):
        #     psi1 = p_m(psi)
        #     psi1 = p_s(psi1)
        # print('done with ER')
        for k in range(j):
            psi1 = iterative_cycle(psi1)
            test_error = rms_intensities(psi1)
            print(test_error)
        print('done with DM :',(b+1),' done')
    psi1 = p_s(f_m(psi1))
    if volume_support == 1:
        psi1 = p_s_vol(psi1)
    return psi1

print('this is a good sign')
psi = np.random.random([2,n,n])*np.exp(0j)*1e-7
c1 = np.zeros_like(I)*np.exp(0j)
D = np.zeros_like(psi)*np.exp(0j)
for j in range(2):
    c1 += 0.5*fftn(psi[j])
C1 = fftn(c1)*mask_array
for j in range(2):
    D[j] = psi[j] + ifftn(C1) - c1
psi = D.copy()

psi = cycle_paralleler(psi)
psi = p_s(f_m(psi))
new_I = abs(fftn(0.5*(psi[0]+psi[1])))**2
psi[1] = focus(psi[1],inverse_phase)
plt.subplot(131)
plt.imshow(abs(psi[0]))
plt.subplot(132)
plt.imshow(abs(psi[1]))
plt.subplot(133)
plt.imshow(fftshift(new_I),cmap='viridis')
plt.show()
q

#
# if __name__ == '__main__':
#     psi_array = np.random.random([c,2,n,n])*np.exp(0j+0)
#     for i in range(c):
#         c1 = np.zeros_like(I)*np.exp(0j+0)
#         for j in range(2):
#             c1 += 0.5*fftn(psi_array[i,j])
#         C1 = fftn(c1)
#         C1 *= mask_array
#         for j in range(2):
#             psi_array[i,j] = psi_array[i,j]+ifftn(C1)-c1
#     times = 3
#     error1 = []
#     error2 = []
#     error3 = []
#     avg_error = []
#     for h in range(times):
#         pool = Pool(processes=1,maxtasksperchild=1)              # start 3 worker processes
#         psi_array = pool.map(cycle_paralleler,(psi_array))
#         pool.close()
#         pool.join()
#
#         error1.append(rms_intensities(psi_array[0]))
#         error2.append(rms_intensities(psi_array[1]))
#         error3.append(rms_intensities(psi_array[2]))
#         avg_error.append((error1[h]+error2[h]+error3[h])/3)
#         print('The average error is ',avg_error[h],' after ',(h+1),' times.')
#
#     plt.figure(1)
#     plt.plot(np.linspace(0,times-1,times),avg_error)
#     psi1 = psi_array[0]
#     psi2 = psi_array[1]
#     psi3 = psi_array[2]
#     psi1 = p_s(f_m(psi1))
#     psi2 = p_s(f_m(psi2))
#     psi3 = p_s(f_m(psi3))
#     psi_average = (psi1+psi2+psi3)/3
#     new_I1 = abs(fftn(psi1))**2
#     new_I2 = abs(fftn(psi2))**2
#     new_I3 = abs(fftn(psi3))**2
#     print(rms_intensities(psi1))
#     print(rms_intensities(psi2))
#     print(rms_intensities(psi3))
#     plt.figure(2)
#     plt.subplot(231)
#     plt.imshow(abs(psi1),cmap='gray')
#     plt.subplot(232)
#     plt.imshow(abs(psi2),cmap='gray')
#     plt.subplot(233)
#     plt.imshow(abs(psi3),cmap='gray')
#     plt.subplot(234)
#     plt.imshow(abs(fftshift(new_I1)),cmap='viridis')
#     plt.subplot(235)
#     plt.imshow(abs(fftshift(new_I2)),cmap='viridis')
#     plt.subplot(236)
#     plt.imshow(abs(fftshift(new_I3)),cmap='viridis')
#     plt.show()
#
#
#
# # print(x_array)
