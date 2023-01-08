package comp3204.classifiers;

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
    }

    //Returns an arraylist of the different classification results along with their confidence level
    public ArrayList getClassConfidence(FImage i){
        org.openimaj.experiment.evaluation.classification.ClassificationResult result = knn.classify((KNNClassifier.imageResize(i)));
        ArrayList confidenceList = new ArrayList();
        for (Object x:result.getPredictedClasses()){
            String str = x.toString().concat(": ").concat(String.valueOf(result.getConfidence(x)));
            confidenceList.add(str);
        }
        return confidenceList;
    }

    //Returns the class with the highest confidence as determined by the kNN Annotator
    public String getHighestConfidentClass(FImage i){
        ClassificationResult result = knn.classify((KNNClassifier.imageResize(i)));
        //Initialises an empty string and confidence value of 0
        String classIdentified = "";
        double confidence = 0;

        //If k in kNN = 1, then the resulting Set of predicted classes will be of size 1 and therefore it returns the class inside the Set
        if (result.getPredictedClasses().size() == 1){
            return result.getPredictedClasses().toString().replace("[", "").replace("]", "");
            //When k > 1, the set has multiple classes identified with different confidence rates, so
        }else{
            //We loop through the set
            for (Object x:result.getPredictedClasses()){
                //Sets the classIdentified string and confidence double to the first element in the set when they're both untouched
                if (classIdentified.length() == 0 && confidence==0){
                    classIdentified = x.toString();
                    confidence = result.getConfidence(x);
                    //If any of the classes in the set thereafter have a greater confidence, the local vars are updated
                }else if (result.getConfidence(x) > confidence){
                    classIdentified = x.toString();
                    confidence = result.getConfidence(x);
                }
            }
            return classIdentified;
        }
    }

    //Applied the trained annotator to a set of images and classifies them
    public void classify(GroupedDataset<String, ListDataset<FImage>, FImage> testing){
        int counter = 0;
        for (FImage i:testing){
            //(uncomment the following line and comment out the one after if you want to view all the classes with their confidence rates for each image)
            //System.out.println("image"+counter+".jpg" + " " + getHighestConfidentClass(i) + " ---- " + getClassConfidence(i));
            System.out.println("image"+counter+".jpg" + " " + getHighestConfidentClass(i));
            counter++;
        }
    }
    public KNNAnnotator getKnn() {
        return knn;
    }
}
