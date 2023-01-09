package comp3204;

import comp3204.classifiers.KNNClassifier;
import comp3204.classifiers.OVAClassifier;
import comp3204.utility.Data;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;

public class Run2 {
    public static void main( String[] args ) throws IOException, URISyntaxException {
        GroupedDataset<String, VFSListDataset<FImage>, FImage> training = Data.training();
        VFSListDataset<FImage> testing = Data.testing();

        OVAClassifier ovaClassifier = new OVAClassifier();

        GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<>(training, 50, 0, 50);
        GroupedDataset<String, ListDataset<FImage>, FImage> trainingSubset = splits.getTrainingDataset();
        GroupedDataset<String, ListDataset<FImage>, FImage> testingSubset = splits.getTestDataset();

//        ovaClassifier.train(trainingSubset);
//        ovaClassifier.classifyOnTrainingData(testingSubset);
        ovaClassifier.train(training);
        ovaClassifier.classify(testing);

//        System.out.println("Evaluation report");
//        ClassificationEvaluator<CMResult<String>, String, FImage> classificationEvaluator =
//                new ClassificationEvaluator<>(
//                        ovaClassifier.getOVA(), testingSubset, new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));
//        System.out.println(classificationEvaluator.analyse(classificationEvaluator.evaluate()));
    }
}
