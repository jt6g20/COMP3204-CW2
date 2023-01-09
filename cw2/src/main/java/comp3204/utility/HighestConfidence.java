package comp3204.utility;

import comp3204.classifiers.KNNClassifier;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.image.FImage;

public class HighestConfidence {
    /**
     * Takes in an FImage to retrieve its classifications' highest confidence class
     * @param result the result of a classification, containing predicted classes and confidence levels
     * @return the class with the highest confidence as determined by the classifier(s)
     */
    public static String getHighestConfidenceClass(ClassificationResult result){
        //Initialises an empty string and confidence value of 0
        String classIdentified = "";
        double highestConfidence = 0;

        if (result.getPredictedClasses().size() == 0) {
            return "none";
        } else {
            for (Object x : result.getPredictedClasses()){
                if (result.getConfidence(x) > highestConfidence){
                    classIdentified = x.toString();
                    highestConfidence = result.getConfidence(x);
                }
            }
            return classIdentified;
        }
    }
}
