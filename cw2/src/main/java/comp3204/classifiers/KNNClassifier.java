package comp3204.classifiers;

import comp3204.utility.HighestConfidence;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.FloatFVComparison;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.ml.annotation.basic.KNNAnnotator;

import java.util.ArrayList;

public class KNNClassifier {
    KNNAnnotator knn;

    /**
     * Takes in an FImage as the argument, crops it into a square about the centre and resizes it
     * @param image you want to resize
     * @return the cropped 16x16px image
     */
    public static FImage imageResize(FImage image){
        //Crops the image about the centre into a square shape with a 1:1 aspect ratio
        int imageCrop = Math.min(image.width, image.height);
        FImage croppedImage = image.extractCenter(imageCrop,imageCrop);
        //Resizes this image down to a 16x16 image
        return croppedImage.processInplace(new ResizeProcessor(16,16));
    }

    /**
     * Takes in an FImage as the argument, turns its pixel value matrix into a 1D vector
     * @param image to be turned into a feature vector (FV)
     * @return feature vector of this 1D image vector
     */
    //Packs all the pixel values into a single vector by concatenating the image rows and returns it as a feature vector
    public static FloatFV concatImgRowsToFV(FImage image){
        return new FloatFV(image.getPixelVectorNative(new float[image.getWidth() * image.getHeight()]));
    }

    /**
     * Takes in the training dataset and trains the kNN annotator based on a k-value
     * @param training dataset
     */
    public void train(GroupedDataset<String, ListDataset<FImage>, FImage> training){
        //the k value from kNN
        int kVal = 1;
        //Feature extractor implementation to pass on into the KNN Annotator
        FeatureExtractor<FloatFV, FImage> extractor = new FeatureExtractor<FloatFV, FImage>() {

            @Override
            public FloatFV extractFeature(FImage image) {
                return (KNNClassifier.concatImgRowsToFV(KNNClassifier.imageResize(image)));
            }
        };
        knn = KNNAnnotator.create(extractor, FloatFVComparison.EUCLIDEAN,kVal);
        knn.trainMultiClass(training);
    }

    /**
     * Takes in an FImage in order to be able to make a list of the image's classifications and their confidences
     * @param i image to get the confidence list of
     * @return arraylist of the different classification results along with their confidence level
     */
    public ArrayList getClassConfidence(FImage i){
        ClassificationResult result = knn.classify((KNNClassifier.imageResize(i)));
        ArrayList confidenceList = new ArrayList();
        for (Object x:result.getPredictedClasses()){
            String str = x.toString().concat(": ").concat(String.valueOf(result.getConfidence(x)));
            confidenceList.add(str);
        }
        return confidenceList;
    }

    /**
     * Applied the trained annotator to a set of images and classifies them
     * @param testing test set
     */
    public void classify(GroupedDataset<String, ListDataset<FImage>, FImage> testing){
        int counter = 0;
        for (FImage i:testing){
            //(uncomment the following line and comment out the one after if you want to view all the classes with their confidence rates for each image)
            //System.out.println("image"+counter+".jpg" + " " + getHighestConfidentClass(i) + " ---- " + getClassConfidence(i));
            System.out.println("image"+counter+".jpg" + " " + HighestConfidence.getHighestConfidenceClass(knn.classify((KNNClassifier.imageResize(i)))));
            counter++;
        }
    }
    public KNNAnnotator getKnn() {
        return knn;
    }
}
