package comp3204;

import comp3204.classifiers.KNNClassifier;
import comp3204.classifiers.OVAClassifier;
import comp3204.utility.Data;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

import java.io.File;
import java.net.URISyntaxException;

public class Run2 {
    public static void main( String[] args ) throws FileSystemException, URISyntaxException {
        GroupedDataset<String, VFSListDataset<FImage>, FImage> training = Data.training();

        System.out.println("Training set size: " + training.size());
        System.out.println("Training set classes:" + training.getGroups());

        OVAClassifier ovaClassifier = new OVAClassifier();

        GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String, FImage>(training, 50, 0, 50);
        GroupedDataset<String, ListDataset<FImage>, FImage> trainingSubset = splits.getTrainingDataset();
//        GroupedDataset<String, ListDataset<FImage>, FImage> testingSubset = splits.getTestDataset();

//        FImage testImage = training.get("bedroom").get(1);
//        System.out.println(testImage.width);
//        System.out.println(testImage.height);
//        System.out.println(ovaClassifier.extractPatchVectors(testImage).length);

        ovaClassifier.train(trainingSubset);
//        ovaClassifier.classify(testingSubset);
    }
}
