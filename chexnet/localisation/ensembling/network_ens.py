import cv2
import numpy as np
from glob import glob
import sys
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import csv
from statistics import stdev
import os
from random import shuffle

# Returns a dictionary mapping file name (without extension) to bounding box
def get_truth_boxes(target_class, truth_file, original_size, new_size):
    truth = {}
    with open(truth_file, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for line in reader:
            fn = line[0].split('.')[0]
            label = line[1]
            if (label == target_class):
                x, y, w, h = line[2:]
                x, y, w, h = int(float(x)), int(float(y)), int(float(w)), int(float(h))
                truth[fn] = [x*new_size//original_size, y*new_size//original_size, w*new_size//original_size, h*new_size//original_size]

    return truth

def largest_cc(im):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(im)
    sizes = stats[:, -1]

    box = [0, 0, 0, 0]

    if (len(sizes) > 1):
        max_label = 1
        max_size = sizes[1]
        for i in range(2, nb_components):
            if sizes[i] > max_size:
                max_label = i
                max_size = sizes[i]

        max_item = stats[max_label]
        box = [max_item[cv2.CC_STAT_LEFT], max_item[cv2.CC_STAT_TOP],
               max_item[cv2.CC_STAT_WIDTH], max_item[cv2.CC_STAT_HEIGHT]]
        
    return box

def get_predicted_boxes(mask_path, thresh):
    thresh = int(thresh*255)
    predictions = {}
    for item in glob(mask_path + '/*'):
        fn = '_'.join(item.split('\\')[-1].split('.')[0].split('_')[:2])
        im = cv2.imread(item,0)
        im[im > thresh] = 255
        im[im < 255] = 0
        x, y, w, h = largest_cc(im)
        #im = cv2.rectangle(im, (x,y), (x+w,y+h), 255, 3)
        predictions[fn] = [x, y, w, h]
        #cv2.imshow('im', im)
        #cv2.waitKey(0)
    return predictions

# Intersection over Union for two bounding boxes
def IOU(a, b):
    Ax1 = a[0]
    Ay1 = a[1]
    Ax2 = Ax1 + a[2]
    Ay2 = Ay1 + a[3]

    Bx1 = b[0]
    By1 = b[1]
    Bx2 = Bx1 + b[2]
    By2 = By1 + b[3]

    intersection = max(0, min(Ax2, Bx2) - max(Ax1, Bx1)) * max(0, min(Ay2, By2) - max(Ay1,By1))
    union = a[2]*a[3] + b[2]*b[3] - intersection

    return float(intersection)/float(union)


def get_top_5_predictions(mask_dir):
    predictions = {}
    for fn in glob(mask_dir + '/*.png'):
        #top_5 = os.path.split(fn)[-1].split('.')[0].split('_')[2:7]
        top_5 = os.path.split(fn)[-1].split('.')[0].split('_')[2:3]
        for i in range(len(top_5)):
            top_5[i] = int(top_5[i])
        fn = '_'.join(os.path.split(fn)[-1].split('_')[:2])
        predictions[fn] = top_5

    return predictions


CLASS_NAME = 'Cardiomegaly'
csv_file = '../data/BBox_List_2017.csv'
all_ims = '../data/localisation_classes_256\\' + CLASS_NAME


gradcam = '..\\results\\' + CLASS_NAME + '\\gradcam\\repeat_'
mp = '..\\results\\' + CLASS_NAME +'\\mp\\repeat_'
im_size = 224

truth_boxes = get_truth_boxes(CLASS_NAME, csv_file, 1024, im_size)

repeats = int(input("How many networks are you ensembling?"))

gc_t = 0.5
mp_t = 0.5

gc_weight = 1
mp_weight = 1

#for gc_t in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#    for mp_t in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#        for gc_weight in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
#            for mp_weight in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:

valid_fns = list(truth_boxes.keys())
shuffle(valid_fns)
valid_fns = valid_fns[:len(valid_fns)//4]
print("Generated sample of length %d" % (len(valid_fns)//4))

#for gc_t in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#    for mp_t in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
for gc_t in [0.5]:
    for mp_t in [0.5]:
        for gc_weight in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            for mp_weight in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
                if gc_weight == 0 and mp_weight == 0:
                    continue
                predictions = []
                gradcam_boxes = []
                mp_boxes = []
                for repeat in range(repeats):
                    curr_gc_fn = gradcam + str(repeat+1)
                    curr_mp_fn = mp + str(repeat+1)
                    predictions.append(get_top_5_predictions(curr_gc_fn))
                    gradcam_boxes.append(get_predicted_boxes(curr_gc_fn, gc_t))
                    mp_boxes.append(get_predicted_boxes(curr_mp_fn, mp_t))


                im_boxes = {}
                for im_fn in valid_fns:
                    curr_gc_boxes = []
                    curr_mp_boxes = []
                    
                    result_box = [0, 0, 0, 0]

                    for repeat in range(repeats):
                        curr_gc_boxes.append(gradcam_boxes[repeat][im_fn])
                        curr_mp_boxes.append(mp_boxes[repeat][im_fn])


                        result_box[0] += (gc_weight*gradcam_boxes[repeat][im_fn][0] + mp_weight*mp_boxes[repeat][im_fn][0])/(gc_weight+mp_weight)
                        result_box[1] += (gc_weight*gradcam_boxes[repeat][im_fn][1] + mp_weight*mp_boxes[repeat][im_fn][1])/(gc_weight+mp_weight)
                        result_box[2] += (gc_weight*gradcam_boxes[repeat][im_fn][2] + mp_weight*mp_boxes[repeat][im_fn][2])/(gc_weight+mp_weight)
                        result_box[3] += (gc_weight*gradcam_boxes[repeat][im_fn][3] + mp_weight*mp_boxes[repeat][im_fn][3])/(gc_weight+mp_weight)

                        
                    for i in range(len(result_box)):
                        result_box[i] /= repeats 
                        result_box[i] = int(result_box[i])

                    im_boxes[im_fn] = result_box



                IOUs = []
                for item in im_boxes:
                    # draw ensemble as blue
                    im = cv2.imread(all_ims + '/' + item + '.png')
                    x, y, w, h = im_boxes[item]
                    im = cv2.rectangle(im, (x,y), (x+w, y+h), (255,0,0), 3)

                    # draw all others are thin
                    for repeat in range(repeats):
                        x, y, w, h = gradcam_boxes[repeat][item]
                        im = cv2.rectangle(im, (x,y), (x+w, y+h), (0,127,127), 1)
                        x, y, w, h = mp_boxes[repeat][item]
                        im = cv2.rectangle(im, (x,y), (x+w, y+h), (0,127,127), 1)

                    # draw truth as green
                    x, y, w, h = truth_boxes[item]
                    im = cv2.rectangle(im, (x,y), (x+w, y+h), (0,255,0), 3)

                    #if t == 0.5:
                    #    cv2.imshow('im', im)
                    #    cv2.waitKey(0)

                    IOUs.append(IOU(im_boxes[item], truth_boxes[item]))
                    #break

                counts = {}
                iou_thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                for item in IOUs:
                    for thresh in iou_thresholds:
                        if thresh not in counts:
                            counts[thresh] = 0 # just so everything exists in the dict
                        if (item >= thresh):
                            if thresh not in counts:
                                counts[thresh] = 1
                            else:
                                counts[thresh] += 1
                    
                # Print errors at each IOU threshold
                errors = []
                for item in iou_thresholds:
                    error = 1 - (counts[item]/len(IOUs))
                    if item == 0.5:
                        #results.append([w1, error])
                        print("%.2f,%.2f,%.2f,%.2f:\t%.3f" % (gc_t, mp_t, gc_weight, mp_weight, error))
                    #print("%.2f:\t%.3f" % (item, error))
                    errors.append(error)

                # Print mean statistics
                #print("Mean IOU: %.3f" % (sum(IOUs)/len(IOUs)))
                #print("SDEV IOU: %.3f" % (stdev(IOUs)))
                sizes = []
                truth_sizes = []
                widths = []
                truth_widths = []
                heights = []
                truth_heights = []
                for item in im_boxes:
                    real_h, real_w = [im_size, im_size]
                    x, y, w, h = im_boxes[item]
                    sizes.append(float(w*h) / (real_w*real_h))
                    widths.append(float(w)/float(real_w))
                    heights.append(float(h)/float(real_h))

                    x, y, w, h = truth_boxes[item]
                    truth_sizes.append(float(w*h) / (real_w*real_h))
                    truth_widths.append(float(w)/real_w)
                    truth_heights.append(float(h)/real_h)
                #print("Area\tWidth\tHeight")
                #print("%.2f\t%.2f\t%.2f" % (sum(sizes)/len(sizes), sum(widths)/len(widths), sum(heights)/len(heights)))
                #print("Truth:")
                #print("%.2f\t%.2f\t%.2f" % (sum(truth_sizes)/len(truth_sizes), sum(truth_widths)/len(truth_widths), sum(truth_heights)/len(truth_heights)))
