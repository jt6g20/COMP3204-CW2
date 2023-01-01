package comp3204.classifiers;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.FloatFVComparison;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.ml.annotation.basic.KNNAnnotator;

public class KNNClassifier {

    public static FImage imageResize(FImage image){
        //Crops the image about the centre into a square shape with a 1:1 aspect ratio
        int imageCrop = Math.min(image.width, image.height);
        FImage croppedImage = image.extractCenter(imageCrop,imageCrop);
        //Resizes this image down to a 16x16 image
        return croppedImage.processInplace(new ResizeProcessor(16,16));
    }

    //Packs all the pixel values into a single vector by concatenating the image rows and returns it as a feature vector
    public static FloatFV concatImgRowsToFV(FImage image){
        return new FloatFV(image.getPixelVectorNative(new float[image.getWidth() * image.getHeight()]));
    }

    //Trains the KNN Annotator
    public void classify(GroupedDataset<String, VFSListDataset<FImage>, FImage> training){
        //the k value from kNN
        int kVal = 1;
        //Feature extractor implementation to pass on into the KNN Annotator
        FeatureExtractor<FloatFV, FImage> extractor = new FeatureExtractor<FloatFV, FImage>() {

            @Override
            public FloatFV extractFeature(FImage image) {
                return KNNClassifier.concatImgRowsToFV(KNNClassifier.imageResize(image));
            }
        };

        KNNAnnotator knn = KNNAnnotator.create(extractor, FloatFVComparison.EUCLIDEAN,kVal);
        knn.trainMultiClass(training);
        int counter = 0;
        for (FImage i:training){
            System.out.println("image"+counter+ ": " + knn.classify(i).getPredictedClasses() + "    " + DisplayUtilities.display(i));
            counter++;
        }


        /*// Classify the test images
        for (FImage image : training) {
            String predictedClass = knn.classify(image).getAnnotations().iterator().next();
            System.out.println(image.getName() + " " + predictedClass);
        }*/
    }
}
