package comp3204.classifiers;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.FloatFVComparison;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.ml.annotation.basic.KNNAnnotator;

public class KNNClassifier {
    KNNAnnotator knn;
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

        /*int counter = 0;
        for (FImage i:training){

            System.out.println("image"+counter+".jpg" + " " + knn.classify((KNNClassifier.imageResize(i))).getPredictedClasses().toString().replace("[", "").replace("]", "") + " ||| " + DisplayUtilities.display(i));
            counter++;
            if (counter==5){
                break;
            }
        }*/
    }

    public void classify(GroupedDataset<String, ListDataset<FImage>, FImage> testing){
        int counter = 0;
        for (FImage i:testing){

            System.out.println("image"+counter+".jpg" + " " + knn.classify((KNNClassifier.imageResize(i))).getPredictedClasses().toString().replace("[", "").replace("]", ""));
            /*System.out.println("image"+counter+".jpg" + " " + knn.classify((KNNClassifier.imageResize(i))).getPredictedClasses().toString().replace("[", "").replace("]", "") + " ||| " + DisplayUtilities.display(i));
            counter++;
            if (counter==5){
                break;
            }*/
        }
    }

    /*public void classify(VFSListDataset<FImage> testing){
        int counter = 0;
        for (FImage i:testing){
            System.out.println("image"+counter+ ": " + knn.classify(i).getPredictedClasses());
            counter++;
        }
    }*/

    public KNNAnnotator getKnn() {
        return knn;
    }
}
