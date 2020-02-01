import torch
from torch.autograd import Variable
import cv2
import numpy as np
from glob import glob

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

np.random.seed(3)
cv2.setRNGSeed(3)

def tv_norm(input, tv_beta):
    img = input[0, 0, :]
    row_grad = torch.mean(torch.abs((img[:-1 , :] - img[1 :, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((img[: , :-1] - img[: , 1 :])).pow(tv_beta))
    return row_grad + col_grad

def preprocess_image(img):
    img = np.float32(img)

    img = img / 255 # normalise

    mean = np.float32(np.array([0.485, 0.456, 0.406]))
    stdev = np.float32(np.array([0.229, 0.224, 0.225]))
    img = (img - mean)/stdev

    img = np.array(np.transpose(img, (2, 0, 1))) # reshape
    preprocessed_img_tensor = torch.from_numpy(img)
    preprocessed_img_tensor.unsqueeze_(0)

    return Variable(preprocessed_img_tensor, requires_grad = False)

def adjust(im):
    return (im - np.min(im)) * ((255 - 0) / (np.max(im) - np.min(im))) + 0

def save(prediction_label, params, fn, predicted_class, mask_size, mask, img, perturbed, output_dir, perturb_type,  iteration=-1, verbose=False, only_show=False):
    mask_size = str(mask_size)
    predicted_class = ''.join(predicted_class.split('_'))
    mask = mask.cpu().data.numpy()[0]
    mask = np.transpose(mask, (1, 2, 0))

    mask = (mask - np.min(mask)) / np.max(mask)
    mask = 1 - mask # So now high value areas correspond to high-distortion areas
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)

    #if (not only_show):
        #cv2.imwrite(predicted_class + '_' + perturb_type + '_' + mask_size + "_heatmap.png", np.uint8(heatmap))
    #    cv2.imwrite("heatmap.png", np.uint8(heatmap))
    
    heatmap = np.float32(heatmap) / 255
    cam = 1.0*heatmap + np.float32(img)/255
    cam = cam / np.max(cam)

    img = np.float32(img) / 255
    perturbed = np.float32(img) / 255

    # mask applied opposite to learning loop, since we inverted it a few lines above
    perturbated = np.multiply(1 - mask, img) + np.multiply(mask, perturbed)

    if only_show == True:
        cv2.imshow('result', np.uint8(heatmap*255))
        cv2.waitKey(1)
    else:
        name = output_dir + '/' + fn + '_'
        for param in params:
           name += str(param) + "_"
        name += '.png'
        #cv2.imwrite(name, np.uint8(255*mask))
        #cv2.imwrite(name + 'x.png', np.uint8(255*cam))
        
        FINAL = np.uint8(255*mask)
        FINAL = adjust(FINAL)

        cv2.imwrite(output_dir + '/' + fn + '_' + str(prediction_label) + '.png', FINAL)
        #print(output_dir + '/' + fn + '_' + str(prediction_label) + '.png')
        #cv2.imwrite('result.png', np.uint8(255*mask))
        
def numpy_to_torch(img, requires_grad = True):
    if len(img.shape) < 3:
        output = np.float32([img])
    else:
        output = np.transpose(img, (2, 0, 1))

    output = torch.from_numpy(output)
    if use_cuda:
        output = output.cuda()

    output.unsqueeze_(0)
    v = Variable(output, requires_grad = requires_grad)
    return v

def load_model(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cpu"):
        model = torch.load(model_name, map_location=lambda storage, loc: storage)
    else:
        model = torch.load(model_name)

    model.eval()
    if use_cuda:
        model.cuda()

    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
            p.requires_grad = False

    return model


def generate_mask(fn, max_iterations, perturb_type, im_size, mask_size, tv_beta, learning_rate, l1_coeff, tv_coeff, output_dir):
  actual_fn = fn.split('\\')[-1].split('.')[0]
  params = [max_iterations, perturb_type, mask_size, tv_beta, learning_rate, l1_coeff, tv_coeff]

  # Load image (load, resize, increase precision)
  original_img = np.float32(cv2.imread(fn))
  img = cv2.resize(original_img, (im_size, im_size))
  original_img_resized = img

  # Compute the perturbed image to apply to original image via mask
  if (perturb_type == 'color'):
      avg = np.sum(img) / (img.shape[0]*img.shape[1]*img.shape[2])
      perturbed_numpy = np.float32(avg*np.ones(img.shape, dtype = np.float32))
  elif (perturb_type == 'blur'):
      blurred_img1 = np.float32(cv2.GaussianBlur(np.uint8(img), (11, 11), 5))
      blurred_img2 = np.float32(cv2.medianBlur(np.uint8(img), 11))
      perturbed_numpy = (blurred_img1 + blurred_img2) / 2
  elif (perturb_type == 'noise'):
      perturbed_numpy = 255*np.float32(np.random.normal(size=(img.shape[0], img.shape[1],1)))
      perturbed_numpy = np.float32(cv2.cvtColor(perturbed_numpy, cv2.COLOR_GRAY2BGR))

  # Create the initial empty, downscaled mask
  mask_init = np.ones((mask_size, mask_size), dtype = np.float32)

  # Preprocess the image and the perturb reference
  img = preprocess_image(img)
  perturbed = preprocess_image(perturbed_numpy)
  perturbed = perturbed.to('cuda')

  # Convert mask to torch
  mask = numpy_to_torch(mask_init)
  mask = mask.to('cuda')

  # Prepare the upsample method
  if use_cuda:
      upsample = torch.nn.UpsamplingBilinear2d(size=(im_size, im_size)).cuda()
  else:
      upsample = torch.nn.UpsamplingBilinear2d(size=(im_size, im_size))
  optimizer = torch.optim.Adam([mask], lr=learning_rate)

  # Get the output tensor from the network
  img = img.to('cuda')
  target = model(img) # apply exponential since we're using log softmax

  topk, indices = torch.topk(target, 2)
  indices = indices[0].to('cpu').numpy()
  prediction_label = ""
  for item in indices[:-1]:
    prediction_label += str(item) + "_"
  prediction_label += str(indices[-1])

  # Determine classification category 
  prediction = target[0][TARGET_CATEGORY].item()
  #print("Cardio has output  %.4f" % (prediction))

  for i in range(max_iterations):
      upsampled_mask = upsample(mask)

      # The single channel mask is used with an RGB image, 
      # so the mask is duplicated to have 3 channel,
      upsampled_mask = upsampled_mask.expand(1, 3, upsampled_mask.size(2), upsampled_mask.size(3))

      # Use the mask to perturbated the input image.
      perturbated_input = img * upsampled_mask + perturbed * (1-upsampled_mask)

      noise = np.zeros((im_size, im_size, 3), dtype = np.float32)
      cv2.randn(noise, 0, 0.2)
      noise = numpy_to_torch(noise)
      perturbated_input = perturbated_input + noise
      perturbated_input.to('cuda')
      
      outputs = model(perturbated_input)

      loss = l1_coeff * torch.mean(torch.abs(1 - mask)) + \
             tv_coeff*tv_norm(mask, tv_beta) + \
             outputs[0, TARGET_CATEGORY]

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # Optional: clamping seems to give better results
      mask.data.clamp_(0, 1)

  upsampled_mask = upsample(mask)
  save(prediction_label, params, actual_fn, 'cardio', mask_size, upsampled_mask, original_img_resized, perturbed_numpy, output_dir, perturb_type)



TARGET_CATEGORY = int(input("Target Category:"))
MODEL_NAME = input(".pth file:")
model = load_model(MODEL_NAME)

output_dir = '.'
im_size = 224
mask_size = 8
max_iterations = 100
learning_rate = 0.1
perturb_type = 'noise'

tv_beta = 3.1
tv_coeff = 10
l1_coeff = 0.11

i = 0
for fn in glob('./cardio_images_256/*.png'):
  generate_mask(fn, max_iterations, perturb_type, im_size, mask_size, tv_beta, learning_rate, l1_coeff, tv_coeff, output_dir)
  i += 1
  print(i)

