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
    wavelength = 1.349e-8
    img_input = np.load('centered_image_spheres.npy')
    sleeve_mask = np.load('sleeve_mask_spheres.npy')
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


    qx_min = np.min(qx_mappedtoflat_array)
    qx_max = np.max(qx_mappedtoflat_array)
    qy_min = np.min(qy_mappedtoflat_array)
    qy_max = np.max(qy_mappedtoflat_array)

    qx_new_array,qy_new_array = np.meshgrid(np.linspace(qx_min,qx_max,x), np.linspace(qy_min,qy_max,y))

    print('Interpolating...')
    remapped_image = griddata(q_rotated_mapped, q_sphere_rotated_mappedToFlat_val_vec, (qx_new_array,qy_new_array),method='linear')
    print('Done interpolating')

    remapped_image[isnan(remapped_image) == True] = 0

    print('Now centering...')
    difflist = []
    rolls = []
    #no y shift after remapping, only x
    img_control = remapped_image.copy()
    inv_img = remapped_image.copy()
    for j in range(2):
        inv_img = rot90(inv_img)
    for i in range(80):
        img1 = np.roll(inv_img,-(i+1))
        difflist.append(norm((img1 - remapped_image)*sleeve_mask))
        rolls.append(-(i+1))
    best = np.argmin(difflist)
    roll = rolls[best]
    shifted_remapped_image = np.roll(remapped_image,-int(roll/2))
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
    for j in range(2):
        flipped_image = rot90(flipped_image)

    #temp is mix of both masks, take difference of resulting image and its inversion
    #to find most centrosymmetric result
    difference = (shifted_remapped_image - flipped_image)*sleeve_mask
    rotate_diff = norm(difference)/norm(shifted_remapped_image)
    #pool_on to test for best distance in parallel, output_packet is both distance
    #and error difference because only a single input and output seem to work with pool.map()
    output_packet = np.array([distance,rotate_diff])
    if pool_on == 1:
        return output_packet
    elif pool_on == 0:
        return shifted_remapped_image


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

volume_support = 0 #to apply volume support once every so often
full_volume = 0 #to apply volume support with every p_s operation
first_time = 0 #for a newly imported image -- subtracts noise and centers
noise = 0 #removes noise on its own -- not recommended if image is already remapped
noise_median = 0 #removes median of left and right halves
noise_poly_fit = 0 #fits a polynomial to the noise and subtracts it out -- also not recommended
second_order = 0
third_order = 1
fourth_order = 0
fifth_order = 0
sixth_order = 0
no_fringe = 0 #crops out center of intensity pattern
visualize = 0#for debugging
ewald_remap = 0 #to remap a 45 degree tilted detector
pool_on = 1 #set to 1 to do a detector distance sweep with remapping
overwrite = 1 #to overwrite files saved to disk
shrinkwrapped = 1 # for implementing a shrinkwrap support later
simulation = 0 #for simulating a reconstruction with no fringes

n = 256

if __name__ == '__main__':
    if first_time == 1:
        img_1 = imread('0008402_20140502_044428.png')
        img_1 = np.roll(img_1,60,axis=0)
        dark = imread('0008402D_20140502_044423.png')
        #subtract out noise:
        scale_factor = np.median(img_1)/np.median(dark)
        img_1 = abs(img_1 - dark*scale_factor)
        img_1 = img_1[24:2024,24:2024]
        p,n = img_1.shape
        fitmask = img_1.copy()
        threshold = 100
        fitmask[img_1 > threshold] = 0
        fitmask[fitmask != 0] = 1
        fitmask[850:1110,880:1150] = 0
        fitmask = ndimage.binary_erosion(fitmask,structure=np.ones([5,5])).astype(np.float32)
        fitmask = ndimage.binary_dilation(fitmask,structure=np.ones([5,5])).astype(np.float32)
        img_1_masked = img_1*fitmask

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
            print(B)
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
            img_1 = img_1 - noisy*scale_factor
            img_1[img_1 < 0] = 0
            img_log = img_1.copy()
            img_log[img_1 != 0] = np.log(img_1[img_1 != 0])
            mask_array = np.ones_like(img_log)
            mask_array = sleeve_mask(beamstop(mask_array))
            mask_array2 = ndimage.binary_erosion(mask_array,structure=np.ones([3,3])).astype(np.float32)
            x_mesh,y_mesh = np.meshgrid(np.arange((-n/2),(n/2)),np.arange((-n/2),(n/2)))
            gaussian = np.exp(-(x_mesh**2 + y_mesh**2))
            mask_array3 = abs(ifftn(fftn(mask_array2)*fftn(gaussian)))
            mask_array3 *= 1/np.max(mask_array3)
            mask_array = mask_array3.copy()
            mask_array[mask_array < 1e-10] = 0
            mask_array[mask_array > 0.75] = 1
            plt.subplot(121)
            plt.imshow(img_log,cmap='viridis')
            plt.title('Natural Log of Noise-Corrected Data Pattern')
            plt.colorbar()
            plt.subplot(122)
            plt.imshow(noisy,cmap='viridis')
            plt.title('3rd Order Polynomial Fit')
            plt.colorbar()
            plt.show()
            plt.imshow(fftshift(np.log(abs(ifftn(fftshift(mask_array)*img_log)))))
            plt.title('Noise Corrected Auto Correlation')
            plt.show()
            q


        #set up zeros around the perimeter:
        sleeve_mask = np.ones([2000,2000])
        sleeve_mask[:,:400] = 0
        sleeve_mask[:,1600:] = 0
        sleeve_mask[:400] = 0
        sleeve_mask[1600:] = 0
        sleeve_mask = transform.resize(sleeve_mask,(p,p))
        img_control = img_1.copy()
        #invert the images and rotate 180 degrees
        inv_img = img_1.copy()
        for j in range(2):
            inv_img = rot90(inv_img)
        #set up error lists
        difflist = []
        difflisty = []
        rolls = []
        rollsy = []
        for i in range(10):
            img1 = np.roll(inv_img,-(i+1))
            #tempxdifflist is appended with the lowest error for each pixel's x coordinate
            tempxdifflist = []
            for j in range(25):
                img2 = np.roll(img1,-(j+1),axis=0)
                difflisty.append(norm((img2-img_1)*sleeve_mask))
                rollsy.append(-(j+1))
                tempxdifflist.append(norm((img2 - img_1)*sleeve_mask))
            difflist.append(np.min(tempxdifflist))
            rolls.append(-(i+1))
            print(i)
        for i in range(10):
            img1 = np.roll(inv_img,(i+1))
            #tempxdifflist is appended with the lowest error for each pixel's x coordinate
            tempxdifflist = []
            for j in range(25):
                img2 = np.roll(img1,-(j+1),axis=0)
                difflisty.append(norm((img2-img_1)*sleeve_mask))
                rollsy.append(-(j+1))
                tempxdifflist.append(norm((img2 - img_1)*sleeve_mask))
            difflist.append(np.min(tempxdifflist))
            rolls.append((i+1))
            print(i)
        #pick out best x and y shifts and roll by half that to centralize
        bestx = np.argmin(difflist)
        xroll = rolls[bestx]
        besty = np.argmin(difflisty)
        yroll = rollsy[besty]
        img_1 = np.roll(img_1,-int(xroll/2))
        img_1 = np.roll(img_1,-int(yroll/2),axis=0)

        if overwrite == 1:
            np.save('centered_image_spheres_nofit',img_1)

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
        img_1 = np.load('centered_image_spheres_nofit.npy')
        p = 256
        img_1 = transform.resize(img_1,(p,p))
        scope = 0.1
        distance = 70e-3
        wavelength = 1.349e-8
        d = 6
        distances = np.linspace((1-scope)*distance,(1+scope)*distance,d)
        a,b = img_1.shape
        N = a # Number of pixels
        du = (2048/N) * 13.5e-6 # pixel size (m)
        pool = Pool(processes=2,maxtasksperchild=1)
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
        best_image = remap_parallel(best_distance)
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
            np.save('remappedoutput_spheres',best_image)

        plt.figure(1)
        plt.imshow(np.log(best_image),cmap='gray')
        plt.title('Remapped Image')
        plt.show()
        q


if __name__ == '__main__':
    n = 1024
    z = 0
    k = 2*np.pi/13.49e-8
    k_x_mesh, k_y_mesh = np.meshgrid(np.arange(int(-n/2),int(n/2)),np.arange(int(-n/2),int(n/2)))
    #propagate fourier space array forward by z length units
    arg = argue(z)
    phase_factor = phase(arg)
    inverse_phase = np.conj(phase_factor)
    # I = np.load('remappedoutput_spheres.npy')
    I = np.load('centered_image_spheres_nofit.npy')
    I_original = imread('0008402_20140502_044428.png')
    I = transform.resize(I,(n,n))
    I_original = transform.resize(I_original,(n,n))
    I[I < 0] = 1
    I[isnan(I) == True] = 1

    mask_array = np.ones_like(I)
    mask_array = sleeve_mask(beamstop(mask_array))
    mask_array[isnan(I) == True] = 0
    mask_array2 = ndimage.binary_erosion(mask_array,structure=np.ones([3,3])).astype(np.float32)
    x_mesh,y_mesh = np.meshgrid(np.arange((-n/2),(n/2)),np.arange((-n/2),(n/2)))
    gaussian = np.exp(-(x_mesh**2 + y_mesh**2))
    mask_array3 = abs(ifftn(fftn(mask_array2)*fftn(gaussian)))
    mask_array3 *= 1/np.max(mask_array3)
    mask_array = mask_array3.copy()
    mask_array[mask_array < 1e-10] = 0
    mask_array[mask_array > 0.75] = 1
    I = ifftshift(I)
    I_original = ifftshift(I_original)
    auto = ifftn(ifftshift(I*mask_array))

    if visualize == 1:
        plt.figure(1)
        plt.subplot(221)
        plt.imshow(fftshift(np.log(I_original)),cmap='viridis')
        plt.title('Original Intensities')
        plt.colorbar()
        plt.subplot(222)
        plt.imshow(fftshift(np.log(abs(ifftn(fftshift(mask_array*I_original))))))
        plt.title('Original Auto-Correlation')
        plt.colorbar()
        plt.subplot(223)
        plt.imshow(fftshift(np.log(I)),cmap='viridis')
        plt.title('Noise-Corrected Intensities')
        plt.colorbar()
        plt.subplot(224)
        plt.imshow(fftshift(np.log(abs(ifftn(fftshift(mask_array*I))))))
        plt.title('Noise-Corrected Auto-correlation')
        plt.colorbar()
        plt.show()
        q


    wavelength = 1.349e-08
    k = 2*np.pi/wavelength
    c = 1
    beta = 0.9
    alphaM = 1/beta
    alphaS = -1/beta
    s = np.zeros([3,1024,1024])
    s[0,200:500,:400] = 1
    s[1,:300,(1024-400):] = 1
    s[2,(1024-300):,(1024-500):(1024-100)] = 1
    s = transform.resize(s,(3,n,n))
    s0 = s.copy()


    if shrinkwrapped == 1:
        Conv_kernel = np.zeros_like(s[0])
        Conv_kernel[510:514,510:514] = 1
        threshold = 0.75

        def shrinkwrap(psi):
            # Do the convolution with FFTs
            psi_mod = np.zeros_like(psi)
            for i in range(len(psi)):
                psi_mod[i] = fftshift(np.abs(ifftn(fftn(psi[i])*Conv_kernel)))
            s_new = np.zeros_like(psi)
            # Make everything larger than a threshold 1 and less than or equal to the threshold zero
            for i in range(len(psi)):
                s_temp = np.zeros_like(s_new[i])
                s_temp[psi_mod[i] > (threshold*np.max(abs(psi_mod[i])))] = 1
                s_new[i] = s_temp
            print(np.sum(s_new[0]),np.sum(s_new[1]),np.sum(s_new[2]))
            plt.subplot(231)
            plt.imshow(abs(s_new[0]))
            plt.subplot(232)
            plt.imshow(abs(s_new[1]))
            plt.subplot(233)
            plt.imshow(abs(s_new[2]))
            plt.subplot(234)
            plt.imshow(abs(s[0]))
            plt.subplot(235)
            plt.imshow(abs(s[1]))
            plt.subplot(236)
            plt.imshow(abs(s[2]))
            plt.show()
            # q
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
        for j in range(len(psi)):
            c1 += psi[j]/len(psi)
        C1 = fftn(c1)
        C1[mask_array == 1] = mags[mask_array == 1]*np.exp(1j*np.angle(C1[mask_array == 1]))
        B = ifftn(C1)
        for j in range(len(psi)):
            D[j] = psi[j]+B-c1
        # C1 = fftn(psi)
        # C1[mask_array == 1] = mags[mask_array == 1]*np.exp(1j*np.angle(C1[mask_array == 1]))
        # D = ifftn(C1)
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
        c = np.zeros_like(psi[0])
        for i in range(len(psi)):
            c += psi[i]/len(psi)
        C1 = fftn(c)
        I1 = abs(C1)**2
        error = np.sqrt(np.sum((I1[mask_array == 1]-I[mask_array == 1])**2)/np.sum(I[mask_array == 1]**2))
        return error

    def cycle_paralleler(psi):
        """Packaged into a function to allow parallelization"""
        i = 20
        j = 1000
        global s
        test_error = rms_intensities(psi)
        print(test_error)
        psi1 = psi.copy()
        for b in range(i):
            # for k in range(j):
            #     psi1 = p_m(psi1)
            #     s = shrinkwrap(psi1)
            #     print(norm(s-s0)/norm(s0))
            #     psi1 = p_s(psi1)
            #     print(rms_intensities(psi1))
            # print('done with ER')
            for k in range(j):
                psi1 = iterative_cycle(psi1)
                test_error = rms_intensities(psi1)
                print(test_error)
            psi1 = p_s(f_m(psi1))
            if shrinkwrapped == 1:
                s = shrinkwrap(psi1)
            print('done with DM :',(b+1),' done')
            c = np.zeros_like(psi[0])
            for i in range(len(psi1)):
                c += abs(psi1[i])/len(psi1)
            plt.imshow(abs(c),cmap='viridis')
            plt.show()
        if volume_support == 1:
            psi1 = p_s_vol(psi1)
        return psi1

    print('this is a good sign')
    psi = np.random.random([3,n,n])*np.exp(0j)*1e-3

    psi = cycle_paralleler(psi)
    psi = p_s(f_m(psi))
    c = np.zeros_like(psi[0])
    for i in range(len(psi)):
        c += psi[i]/len(psi)
    C1 = fftn(c)
    new_I = abs(C1)**2
    plt.subplot(121)
    plt.imshow(abs(c))
    plt.title('Reconstructed Image')
    plt.subplot(122)
    plt.imshow(fftshift(new_I),cmap='viridis')
    plt.title('Reconstructed Intensity')
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
