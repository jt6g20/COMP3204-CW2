package comp3204;

import comp3204.classifiers.PHOWClassifier;
import comp3204.utility.Data;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.image.FImage;

import java.util.concurrent.TimeUnit;

public class Run3 {
    public static void main( String[] args ) throws Exception{

        GroupedDataset<String, VFSListDataset<FImage>, FImage> training = Data.training();
        VFSListDataset<FImage> testing = Data.testing();

        PHOWClassifier phow = new PHOWClassifier();

        GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<>(training, 30, 0, 30);

        GroupedDataset<String, ListDataset<FImage>, FImage> trainingSubset = splits.getTrainingDataset();
        GroupedDataset<String, ListDataset<FImage>, FImage> testingSubset = splits.getTestDataset();

        long trainStart = System.currentTimeMillis();
        phow.train(training);
        long trainEnd = System.currentTimeMillis();
        System.out.println("Training time: " + msToMinSec(trainEnd - trainStart));

        long classifyStart = System.currentTimeMillis();
        phow.classify(testing);
        long classifyEnd = System.currentTimeMillis();
        System.out.println("Classify time: " + msToMinSec(classifyEnd - classifyStart));

//        long evaluateStart = System.currentTimeMillis();
//        System.out.println("____________________\nEvaluation report");
//        ClassificationEvaluator<CMResult<String>, String, FImage> classificationEvaluator =
//                new ClassificationEvaluator<CMResult<String>, String, FImage>(
//                        phow.getAnn(), testingSubset, new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));
//        System.out.println(classificationEvaluator.analyse(classificationEvaluator.evaluate()));
//        long evaluateEnd = System.currentTimeMillis();
//        System.out.println("Evaluate time: " + msToMinSec(evaluateEnd - evaluateStart));
    }

    public static String msToMinSec(long ms){
        long min = (ms / 1000) / 60;
        int sec = (int)(ms / 1000) % 60;
        return min + ":" + sec;
    }
}
