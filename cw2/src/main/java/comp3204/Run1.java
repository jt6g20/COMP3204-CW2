package comp3204;

import comp3204.classifiers.KNNClassifier;
import comp3204.utility.Data;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;

import java.net.URISyntaxException;
import java.util.*;

public class Run1 {
    public static void main( String[] args ) throws FileSystemException, URISyntaxException {
      //The real training and testing datasets
        GroupedDataset<String, VFSListDataset<FImage>, FImage> training = Data.training();
        VFSListDataset<FImage> testing = Data.testing();

        //Instantiates a new KNNClassifier Object from our KNNClassifier class
        /*KNNClassifier knnClassifier = new KNNClassifier();
        //Passes in the training set so the classifier can annotate all the images with their class labels
        knnClassifier.train(training);
        //Runs the classifier on the testing set so it uses the existing data points and the kNN algorithm to classify the images
        knnClassifier.classify(testing);*/

      //For testing purposes only in order to find the accuracy
        KNNClassifier knnClassifier1 = new KNNClassifier(); //New KNNClassifier object
      //Training dataset split into two new smaller equally sized randomly selected subsets: a new training set and a new testing set (used for validation)
        GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String, FImage>(training, 50, 0, 50);
        GroupedDataset<String, ListDataset<FImage>, FImage> trainingSubset = splits.getTrainingDataset();
        GroupedDataset<String, ListDataset<FImage>, FImage> testingSubset = splits.getTestDataset();

        //Trains the dataset on this
        knnClassifier1.train(trainingSubset);
        knnClassifier1.classify(testingSubset);

        //Uses the OpenIMAJ ClassificationEvaluator class to calculate the classification accuracies
        System.out.println("Evaluation report");
        ClassificationEvaluator<CMResult<String>, String, FImage> classificationEvaluator =
            new ClassificationEvaluator<CMResult<String>, String, FImage>(
                knnClassifier1.getKnn(), testingSubset, new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));
        System.out.println(classificationEvaluator.analyse(classificationEvaluator.evaluate()).getDetailReport());

        //The following blocks of code were used for testing purposes only, and left in here in case they need to be used again to test
        //prints all the image classes and their names
        /*int counter = 0;
        for (Object i:trainingSubset.getGroups()){
            System.out.println(trainingSubset.get(i).size());;
            DisplayUtilities.display(trainingSubset.get(i).get(0));
            DisplayUtilities.display(trainingSubset.get(i).get(1));
            counter++;
            if (counter==1){
                break;
            }
        }
        counter = 0;
        for (Object i:testingSubset.getGroups()){
            System.out.println(testingSubset.get(i).size());;
            DisplayUtilities.display(testingSubset.get(i).get(0));
            DisplayUtilities.display(testingSubset.get(i).get(1));
            counter++;
            if (counter==1){
                break;
            }
        }*/

        //System.out.println("SPLIT TRAINING:" + (splits.getTrainingDataset().get(splits.getTrainingDataset().keySet().toArray()[0])) + " SIZE:" + splits.getTrainingDataset().size());
        //System.out.println("SPLIT TESTING:" + splits.getTestDataset().keySet() + " SIZE:" + splits.getTestDataset().size());

        /*
        //prints all the image classes and their names
        for (String i:training.keySet()){
            for(FImage j: training.get(i)){
                j = knn.imageResize(j);
                System.out.println(i + ": " + "size " + j.width + "x" + j.height + DisplayUtilities.display(j));
            }
        }

        System.out.println("Testing set size: " + testing.size());


        // Testing knn image cropping
        KNNClassifier knn = new KNNClassifier();
        FImage croppedImage = knn.imageResize(testing.get(0));
        //DisplayUtilities.display(testing.get(0));
        DisplayUtilities.display(croppedImage);
        float[] packedImgPixelVec = knn.concatImgRowsToVec(croppedImage);

        //The following 4 lines can be safely deleted once they are no longer needed
        System.out.println("Image breakdown:");
        showMatrix(croppedImage.pixels); //Prints 16x16 matrix of the cropped image for diagnosing purposes
        System.out.println("________________________________________________");
        System.out.println(Arrays.toString(packedImgPixelVec)); //Prints out packed img vector*/

    }

    //Prints out a 2D vector in the form of a matrix
    public static void showMatrix(float[][] mat) {
        for (float[] row: mat)
            System.out.println(Arrays.toString(row));
    }

}
