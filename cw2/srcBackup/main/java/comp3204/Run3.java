package comp3204;

import comp3204.classifiers.PHOWClassifier;
import comp3204.utility.Data;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.image.FImage;

import java.net.URISyntaxException;

public class Run3 {
    public static void main( String[] args ) throws FileSystemException, URISyntaxException {

        GroupedDataset<String, VFSListDataset<FImage>, FImage> training = Data.training();
        VFSListDataset<FImage> testing = Data.testing();

        PHOWClassifier phow = new PHOWClassifier();

        GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<>(training, 15, 0, 15);

        GroupedDataset<String, ListDataset<FImage>, FImage> trainingSubset = splits.getTrainingDataset();
        GroupedDataset<String, ListDataset<FImage>, FImage> testingSubset = splits.getTestDataset();

        phow.train(trainingSubset);
        phow.classify(testingSubset);

        System.out.println("Evaluation report");
        ClassificationEvaluator<CMResult<String>, String, FImage> classificationEvaluator =
                new ClassificationEvaluator<>(
                        phow.getAnn(), testingSubset, new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));
        System.out.println(classificationEvaluator.analyse(classificationEvaluator.evaluate()).getDetailReport());
    }
}
