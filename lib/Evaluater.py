
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
import glob
tfkl = tf.keras.layers
import matplotlib.pyplot as plt
import time
#own libs
from lib.load_image import *

from lib.Training import *
from lib.ModelArchitecture import *
#start timer
start_time = time.time()
import datetime
from lib.Visual import *
import imageio
import matplotlib.ticker as mtick




def fullEvaluationImage(model, path, saveFolder, index=0, cutoffValue=0.90, resolution=50):
    image, ground_truth = load_image(path)
    images = [image]
    images = tf.convert_to_tensor(images)
    all_poses = get_all_poses(resolution,resolution,resolution)
    predictions = model.generate_pdf(images, all_poses)
    predictions = np.squeeze(predictions)
    #predictions = predictions / np.max(predictions)
    vals_greater_01 = 1


    loss_value = calculateEvaluationLoss(ground_truth, images, model)

    # combine, sort and break apart
    indices = np.argsort(predictions)
    sortedPredictions=predictions[indices]
    sortedPoses=all_poses[indices]

    cumulativePredictions=np.cumsum(sortedPredictions)

    size = len(cumulativePredictions)
    lowIndex = np.searchsorted(cumulativePredictions, 0.0001)
    precentCutoffIndex = np.searchsorted(cumulativePredictions, 1-cutoffValue)

    percentile = precentCutoffIndex / size
    percent = cumulativePredictions[precentCutoffIndex]


    #=================CUMULATIVE FIGURE====================
    cumulativeFigure = createPlot(cumulativePredictions, precentCutoffIndex, startIndex=lowIndex)

    #================ExtractGround Truth=======================
    gt_x, gt_y, gt_z = extractGroundTruth(ground_truth)


    #=================MAKE PREDICTION=====================
    r_pred, x_pred, y_pred = makePrediction(model, path)

    #=================BoundingBoxMIN-MAX================
    rmaxCutoff, rminCutoff, xmaxCutoff, xminCutoff, ymaxCutoff, yminCutoff = getBoundingBoxLimits(precentCutoffIndex,
                                                                                                  sortedPoses)

    #print(cumulativePredictions)

    boundingBoxFigure = SingleUseHeatmapWithBoundingBox(image, sortedPoses, sortedPredictions,
                                                        gt_z, gt_y, gt_x,
                                                        x_pred,y_pred, r_pred,
                                                        xminCutoff, xmaxCutoff, yminCutoff, ymaxCutoff, rminCutoff, rmaxCutoff)


    #=========================Packing===========================
    percentileInfo = [percentile, percent]
    cutoffLimitBoundingBox=[xminCutoff,xmaxCutoff,yminCutoff,ymaxCutoff,rminCutoff,rmaxCutoff]



    groundTruth=np.array([gt_x,gt_y,gt_z])
    prediction = np.array([x_pred,y_pred,r_pred])

    boundingBoxFigure.savefig(saveFolder+f"Heatmap_BoundingBox_{index}")
    cumulativeFigure.savefig(saveFolder + f"Cumulative_{index}")

    return loss_value, percentileInfo,cutoffLimitBoundingBox,groundTruth,prediction


def SingleUseHeatmapWithBoundingBox(image, sortedPoses, sortedPredictions, gt_z, gt_y, gt_x, x_pred, y_pred, r_pred,
                                    xminCutoff, xmaxCutoff, yminCutoff, ymaxCutoff, rminCutoff, rmaxCutoff):
    boundingBoxFigure = plt.figure(0, figsize=(12, 12))
    boundingBoxFigure.clf()
    figure_2D_ax_xy = boundingBoxFigure.add_subplot(221)
    figure_2D_ax_image = boundingBoxFigure.add_subplot(222)
    figure_3D_ax_heat = boundingBoxFigure.add_subplot(223, projection='3d')
    figure_2D_ax_rot = boundingBoxFigure.add_subplot(224)
    scaledSortedPredictions = sortedPredictions / np.max(sortedPredictions)
    figure_2D_ax_xy.clear()
    figure_2D_ax_image.clear()
    figure_3D_ax_heat.clear()
    figure_2D_ax_rot.clear()
    figure_2D_ax_xy.set_title("Top View", fontsize=20)
    figure_2D_ax_xy.set_xlabel("X")
    figure_2D_ax_xy.set_ylabel("Y")
    figure_3D_ax_heat.set_title("3D Heatmap", fontsize=20)
    figure_3D_ax_heat.set_xlabel("X")
    figure_3D_ax_heat.set_ylabel("Y")
    figure_3D_ax_heat.set_zlabel("Rotation")
    figure_2D_ax_rot.set_title("Side View", fontsize=20)
    figure_2D_ax_rot.set_xlabel("Y")
    figure_2D_ax_rot.set_ylabel("Rotation")
    figure_2D_ax_xy.set_xlim([-0.33, 0.33])
    figure_2D_ax_xy.set_ylim([-0.33, 0.33])
    figure_3D_ax_heat.set_xlim([-0.33, 0.33])
    figure_3D_ax_heat.set_ylim([-0.33, 0.33])
    figure_3D_ax_heat.set_zlim([-3.33, 3.33])
    figure_2D_ax_rot.set_xlim([-0.33, 0.33])
    figure_2D_ax_rot.set_ylim([-3.33, 3.33])
    markerStyle = matplotlib.markers.MarkerStyle('o', fillstyle='none')
    for i in range(np.size(sortedPredictions)):
        # print("Prediction: ", predictions[i])
        pose = sortedPoses[i]
        currentX = pose[0]
        currentY = pose[1]
        currentR = math.atan2(pose[3], pose[2])
        currentX, currentY = unnormalizePose(currentX, currentY)
        if (scaledSortedPredictions[i] >= 0.1):
            # print(i)
            figure_2D_ax_xy.plot([currentX], [currentY], marker='o', markersize=2, color="red",
                                 alpha=np.clip(scaledSortedPredictions[i] - 0.1, 0, 1))  # , label='PP')
            figure_3D_ax_heat.plot([currentX], [currentY], [currentR], marker='o', markersize=2, color="red",
                                   alpha=np.clip(scaledSortedPredictions[i] - 0.1, 0, 1))  # , label='PP')
            figure_2D_ax_rot.plot([currentY], [currentR], marker='o', markersize=2, color="red",
                                  alpha=np.clip(scaledSortedPredictions[i] - 0.1, 0, 1))  # , label='PP')
        else:
            pass
    figure_2D_ax_xy.plot([x_pred], [y_pred], marker=markerStyle, markersize=10, color="blue")  # , label='GT')
    figure_3D_ax_heat.plot([x_pred], [y_pred], [r_pred], marker=markerStyle, markersize=10,
                           color="blue")  # , label='GT')
    figure_2D_ax_rot.plot([y_pred], [r_pred], marker=markerStyle, markersize=10, color="blue")  # , label='GT')
    # getIterativeMaxPose(self, imagePath, resolution, depth=10, zoomFactor=0.5):
    # getMaxPose(self, images , x_num, y_num, r_num, xmin=-0.3, xmax=0.3, ymin=-0.3, ymax=0.3, rmin=0,rmax=360, training=False):
    figure_2D_ax_xy.plot([gt_x], [gt_y], marker=markerStyle, markersize=10, color="green")  # , label='GT')
    figure_3D_ax_heat.plot([gt_x], [gt_y], [gt_z], marker=markerStyle, markersize=10,
                           color="green")  # , label='GT')
    figure_2D_ax_rot.plot([gt_y], [gt_z], marker=markerStyle, markersize=10, color="green")  # , label='GT')
    figure_2D_ax_rot.axhline(rminCutoff)
    figure_2D_ax_rot.axhline(rmaxCutoff)
    figure_2D_ax_rot.axvline(yminCutoff)
    figure_2D_ax_rot.axvline(ymaxCutoff)
    figure_2D_ax_xy.axhline(yminCutoff)
    figure_2D_ax_xy.axhline(ymaxCutoff)
    figure_2D_ax_xy.axvline(xminCutoff)
    figure_2D_ax_xy.axvline(xmaxCutoff)
    figure_2D_ax_image.imshow(image)
    figure_2D_ax_image.set_title("Image", fontsize=20)
    figure_2D_ax_image.axis('off')
    #boundingBoxFigure.canvas.draw()
    #boundingBoxFigure.canvas.flush_events()
    print("Done Plotting 2D plots")
    return boundingBoxFigure


def getBoundingBoxLimits(precentCutoffIndex, sortedPoses):
    posesAboveCutoffPercentage = sortedPoses[precentCutoffIndex:]
    # print(posesAboveCutoffPercentage)
    rotationsAboveCutoffPercentage = np.arctan2(posesAboveCutoffPercentage[:, 3], posesAboveCutoffPercentage[:, 2])


    xmaxCutoff = np.max(posesAboveCutoffPercentage[:, 0])
    ymaxCutoff = np.max(posesAboveCutoffPercentage[:, 1])
    xminCutoff = np.min(posesAboveCutoffPercentage[:, 0])
    yminCutoff = np.min(posesAboveCutoffPercentage[:, 1])

    rminCutoff = np.min(rotationsAboveCutoffPercentage[:])
    rmaxCutoff = np.max(rotationsAboveCutoffPercentage[:])
    rotationsAboveCutoffPercentage[rotationsAboveCutoffPercentage < 0] += 2 * np.pi
    rminCutoff2 = np.min(rotationsAboveCutoffPercentage[:])
    rmaxCutoff2 = np.max(rotationsAboveCutoffPercentage[:])


    if(abs(rminCutoff-rmaxCutoff)>abs(rminCutoff2-rmaxCutoff2)):
        rminCutoff=rminCutoff2
        rmaxCutoff=rmaxCutoff2

    xminCutoff, xmaxCutoff = unnormalizePose(xminCutoff, xmaxCutoff)
    yminCutoff, ymaxCutoff = unnormalizePose(yminCutoff, ymaxCutoff)
    return rmaxCutoff, rminCutoff, xmaxCutoff, xminCutoff, ymaxCutoff, yminCutoff


def makePrediction(model, path):
    pose_guess_vector = model.getIterativeMaxPose(path, 30, 3, 4 / 20)
    x_pred = pose_guess_vector[0]
    y_pred = pose_guess_vector[1]
    r_pred = math.radians(pose_guess_vector[2])
    return r_pred, x_pred, y_pred


def extractGroundTruth(ground_truth):
    gt_x = float(ground_truth["x"])
    gt_y = float(ground_truth["y"])
    gt_z = float(ground_truth["r"])
    gt_z = math.radians(gt_z)
    if gt_z > math.pi:
        gt_z = gt_z - 2 * math.pi
    return gt_x, gt_y, gt_z


def calculateEvaluationLoss(ground_truth, image, model):

    losses = []
    x = ground_truth["x"]
    y = ground_truth["y"]
    r = ground_truth["r"]
    x, y, r = float(x), float(y), float(r)
    x, y = normalizePos(x, y)
    r_rad = math.radians(r)
    newPose = np.array([[x, y, math.cos(r_rad), math.sin(r_rad)]])
    losses=[]
    images = []
    for i in range(4):
        images.append(image[0])
    images = tf.convert_to_tensor(np.array(images))
    for i in range(4):




        poses = np.zeros((4, 100, 4))
        #poses = np.zeros((1, 50*50*50, 4))
        for i in range(1):
            #poses[i] = get_all_poses(50,50,50)
            poses[i]=get_random_poses_plus_correct(100,ground_truth)
            #poses[i][-1]=newPose


        # convert numpy arrays to tensors

        poses = tf.convert_to_tensor(poses)


        logits_norm = model.generate_pdf(images, poses)
        loss_value = -(tf.math.log(logits_norm[0, -1] / (
                ((0.6 ** 2) * 3.1415 * 2) / 100)))  # index -1 because last one is the correct pose
        print(logits_norm[:, -1])
        print(loss_value)
        print(-tf.math.log(1 / (
                ((0.6 ** 2) * 3.1415 * 2) / 100)  ))
        losses.append(loss_value)
    return np.mean(np.array(losses))


def createPlot(plotData, indicatorIndex, startIndex=-1,endIndex=-1):

    size = len(plotData)
    if startIndex == -1: startIndex=0
    if endIndex == -1: endIndex = size
    cumulativeFigure = plt.figure(1, (7, 4))
    cumulativeFigure.clf()
    ax = cumulativeFigure.add_subplot(1, 1, 1)
    ax.axhline(plotData[indicatorIndex])
    ax.axvline(indicatorIndex, label="test")
    # print(percent,percentile)
    ax.plot(plotData)
    # print(np.sum(sortedPredictions[precentCutoffIndex:]))
    # print(lowIndex)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(size))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.set_xlim([startIndex, endIndex])
    #cumulativeFigure.show()
    return cumulativeFigure


def evaluatePictures(model, files, outputFolder,cutoffPercentage=0.9, resolution=50, maxEvaluations=-1):
    if maxEvaluations==-1: maxEvaluations=len(files)

    lossValues=[]
    percentiles=[]
    variances=[]
    predictionDeltas=[]

    numPoses=resolution*resolution*resolution


    for i in range(maxEvaluations):


        print(f"===========================\n"
              f"\tEvaluation: {i}\n"
              f"===========================\n")
        loss_value, percentileInfo, cutoffLimitBoundingBox, groundTruth, prediction=fullEvaluationImage(model,files[i],outputFolder,i,cutoffPercentage,resolution)


        lossValues.append(loss_value)
        percentiles.append(percentileInfo[0])

        varX = cutoffLimitBoundingBox[0]-cutoffLimitBoundingBox[1]
        varY = cutoffLimitBoundingBox[2] - cutoffLimitBoundingBox[3]
        varR = cutoffLimitBoundingBox[4] - cutoffLimitBoundingBox[5]
        variance = np.abs(np.array([varX,varY,varR]))
        variances.append(variance)
        predictionDeltas.append(np.abs(groundTruth-prediction))


    lossValues = np.array(lossValues)
    percentiles = np.array(percentiles)
    variances = np.array(variances)
    predictionDeltas = np.array(predictionDeltas)
    file = open(outputFolder+"data.txt", "w")

    file.write(f"Evaluations := {maxEvaluations};\n")
    file.write(f"CutoffPercentage := {cutoffPercentage};\n")
    file.write(f"Resolution := {resolution};\n")

    file.write("\n")

    file.write( "%Loss\n")
    file.write( "averageLoss := "   +   np.array2string(np.average(lossValues), formatter={'float_kind':lambda x: "%.4f" % x})  +"\n")
    file.write("minLoss := "        +   np.array2string(np.min(lossValues), formatter={'float_kind':lambda x: "%.4f" % x})      +"\n")
    file.write("maxLoss := "        +   np.array2string(np.max(lossValues), formatter={'float_kind':lambda x: "%.4f" % x})      +"\n")

    file.write("\n")

    file.write("%PosesBelow90%\n")
    file.write("averagePercentile := "  + np.array2string(np.average(percentiles)*numPoses, formatter={'float_kind':lambda x: "%.4f" % x})   + "\n")
    file.write("minPercentile := "      + np.array2string(np.min(percentiles)*numPoses, formatter={'float_kind':lambda x: "%.4f" % x})       + "\n")
    file.write("maxPercentile := "      + np.array2string(np.max(percentiles)*numPoses, formatter={'float_kind':lambda x: "%.4f" % x})       + "\n")
    file.write("\n")

    file.write("%PosesAbove90%\n")
    file.write("averagePercentile := " + np.array2string(numPoses-np.average(percentiles) * numPoses,
                                                         formatter={'float_kind': lambda x: "%.4f" % x}) + "\n")
    file.write("maxPercentile := " + np.array2string(numPoses-np.min(percentiles) * numPoses,
                                                     formatter={'float_kind': lambda x: "%.4f" % x}) + "\n")
    file.write("minPercentile := " + np.array2string(numPoses-np.max(percentiles) * numPoses,
                                                     formatter={'float_kind': lambda x: "%.4f" % x}) + "\n")

    file.write("\n")

    file.write("%Percentile\n")
    file.write("averagePercentile := " + np.array2string(np.average(percentiles),
                                                         formatter={'float_kind': lambda x: "%.4f" % x}) + "\n")
    file.write("minPercentile := " + np.array2string(np.min(percentiles),
                                                     formatter={'float_kind': lambda x: "%.4f" % x}) + "\n")
    file.write("maxPercentile := " + np.array2string(np.max(percentiles),
                                                     formatter={'float_kind': lambda x: "%.4f" % x}) + "\n")

    file.write("\n")

    file.write("%Variances\n")


    averageVariance = np.average(variances, 0)
    minVariance = np.min(variances, 0)
    maxVariance = np.max(variances, 0)

    file.write("averageVariance := " + np.array2string(averageVariance, separator=',', formatter={'float_kind': lambda x: "%.4f" % x}) + "\n")
    file.write("minVariance := " + np.array2string(minVariance, separator=',', formatter={'float_kind': lambda x: "%.4f" % x}) + "\n")
    file.write("maxVariance := " + np.array2string(maxVariance, separator=',', formatter={'float_kind': lambda x: "%.4f" % x}) + "\n")

    file.write("\n")

    file.write("%PredictionDeltas\n")
    averageDelta = np.average(predictionDeltas, 0)
    minDelta = np.min(predictionDeltas, 0)
    maxDelta = np.max(predictionDeltas, 0)
    file.write("averageDelta := " + np.array2string(averageDelta, separator=',', formatter={'float_kind':lambda x: "%.4f" % x}) + "\n")
    file.write("minDelta := " + np.array2string(minDelta, separator=',', formatter={'float_kind':lambda x: "%.4f" % x}) + "\n")
    file.write("maxDelta := " + np.array2string(maxDelta, separator=',', formatter={'float_kind':lambda x: "%.4f" % x}) + "\n")

    file.close()