
package comp3204.classifiers;

import comp3204.utility.HighestConfidence;
import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntFloatPair;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class OVAClassifier {
    LiblinearAnnotator<FImage, String> classifier;

    public List<FloatFV> extractPatchVectors(FImage image, int patchSize, int increment){

        ArrayList<FloatFV> vectors = new ArrayList<>();

        int xIncrements = (int) Math.floor((image.getWidth()-patchSize) / increment);
        int yIncrements = (int) Math.floor((image.getHeight()-patchSize) / increment);

        for (int y = 0; y < yIncrements; y++) {
            for (int x = 0; x < xIncrements; x++) {
                FImage sample = image.extractROI(x*increment,y*increment, patchSize, patchSize);
                float[] rawVectorArray = sample.getPixelVectorNative(new float[sample.getWidth() * sample.getHeight()]);
                vectors.add(new FloatFV(rawVectorArray));
            }
        }

        return vectors;
    }

    public HardAssigner<float[], float[], IntFloatPair> trainQuantiser(GroupedDataset<String, ListDataset<FImage>, FImage> sample, int clusters){
        List<FloatFV> vectors = new ArrayList<>();

        //collect feature vectors from images
        System.out.println("collecting feature vectors...");
        for(ListDataset<FImage> list : sample.values()) {
            for (FImage image : list) {
                vectors.addAll(extractPatchVectors(image, 8, 4));
            }
        }

        //limit sample size for training quantiser
        System.out.println(vectors.size());
        if (vectors.size() > 10000) {
            Collections.shuffle(vectors);
            vectors = vectors.subList(0, 10000);
        }
        System.out.println(vectors.size());

        System.out.println("clustering...");
        FloatKMeans cluster = FloatKMeans.createExact(clusters);
        FloatCentroidsResult result = cluster.cluster(vectorListToArray(vectors));
        System.out.println("clustering complete");
        return result.defaultHardAssigner();
    }

    class Extractor implements FeatureExtractor<DoubleFV, FImage> {

        HardAssigner<float[], float[], IntFloatPair> assigner;

        public Extractor(HardAssigner<float[], float[], IntFloatPair> assigner)
        {
            this.assigner = assigner;
        }

        @Override
        public DoubleFV extractFeature(FImage image) {
            BagOfVisualWords<float[]> bovw = new BagOfVisualWords<>(assigner);

            return bovw.aggregateVectors(extractPatchVectors(image, 8, 4)).asDoubleFV();
        }
    }

    public void train(GroupedDataset<String, ListDataset<FImage>, FImage> training){

        //train a quantiser using a random sample of n images from the training dataset
        HardAssigner<float[], float[], IntFloatPair> assigner = trainQuantiser(
                GroupedUniformRandomisedSampler.sample(training, 30), 500);

        FeatureExtractor<DoubleFV, FImage> extractor = new Extractor(assigner);

        classifier = new LiblinearAnnotator<>(
                extractor, LiblinearAnnotator.Mode.MULTILABEL, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);

        System.out.println("training...");
//        stratified sampling used to speed up for development
//        classifier.train(new StratifiedGroupedUniformRandomisedSampler(2).sample(training));
        classifier.train(training);
        System.out.println("training complete");
    }

    public void classify(GroupedDataset<String, ListDataset<FImage>, FImage> testing){
        int count = 0;
        for (String category : testing.keySet()) {
            System.out.println(category + " ----------------------------------------------");
            for (FImage i : testing.get(category)) {
                ClassificationResult result = classifier.classify(i);
                System.out.println(count + ".jpg " + HighestConfidence.getHighestConfidenceClass(result));
                count++;
            }
        }
    }
    public float[][] vectorListToArray(List<FloatFV> list) {
        int vectorSize = list.get(0).length();

        float[][] vectors = new float[list.size()][vectorSize];
        for (int i = 0; i < list.size(); i++) {
            for (int j = 0; j < vectorSize; j++) {
                vectors[i][j] = list.get(i).get(j);
            }
        }

        return vectors;
    }

    public LiblinearAnnotator<FImage, String> getOVA() {
        return classifier;
    }
}
