package comp3204;

import comp3204.classifiers.KNNClassifier;
import comp3204.utility.Data;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;
import java.util.*;

public class Run1 {
    public static void main( String[] args ) throws Exception {

      //The real training and testing datasets
        GroupedDataset<String, VFSListDataset<FImage>, FImage> training = Data.training();
        VFSListDataset<FImage> testing = Data.testing();
        //Instantiates a new KNNClassifier Object from our KNNClassifier class
        KNNClassifier knnClassifier = new KNNClassifier();
        //Passes in the training set so the classifier can annotate all the images with their class labels
        knnClassifier.train(training);
        //Runs the classifier on the testing set, so it uses the existing data points and the kNN algorithm to classify the images
        knnClassifier.classify(testing, "Run1");

      //For testing purposes only in order to find the accuracy
        //KNNClassifier knnClassifier1 = new KNNClassifier(); //New KNNClassifier object
      //Training dataset split into two new smaller equally sized randomly selected subsets: a new training set and a new testing set (used for validation)
        /*GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String, FImage>(training, 50, 0, 50);
        GroupedDataset<String, ListDataset<FImage>, FImage> trainingSubset = splits.getTrainingDataset();
        GroupedDataset<String, ListDataset<FImage>, FImage> testingSubset = splits.getTestDataset();

        //Trains the dataset on this
        knnClassifier1.train(trainingSubset);
        knnClassifier1.classify(testingSubset, "Run1");

        //Uses the OpenIMAJ ClassificationEvaluator class to calculate the classification accuracies
        System.out.println("____________________\nEvaluation report");
        ClassificationEvaluator<CMResult<String>, String, FImage> classificationEvaluator =
            new ClassificationEvaluator<CMResult<String>, String, FImage>(
                knnClassifier1.getKnn(), testingSubset, new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));
        System.out.println(classificationEvaluator.analyse(classificationEvaluator.evaluate()));*/
    }

    //Prints out a 2D vector in the form of a matrix
    public static void showMatrix(float[][] mat) {
        for (float[] row: mat)
            System.out.println(Arrays.toString(row));
    }

}
