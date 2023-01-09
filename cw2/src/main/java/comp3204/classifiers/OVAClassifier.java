
package comp3204.classifiers;

import comp3204.utility.HighestConfidence;
import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.dataset.sampling.StratifiedGroupedUniformRandomisedSampler;
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

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class OVAClassifier {
    LiblinearAnnotator<FImage, String> classifier;

    /**
     * Extracts feature vectors from fixed size densely-sampled pixel patches
     * @param image image to extract features from
     * @param patchSize dimensions of the patches
     * @param increment number of pixels to move patch by for each feature vector
     * @return a list of feature vectors
     */
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

    /**
     * Constructs a HardAssigner trained on the dataset to perform K-means clustering on feature vectors
     * @param images a dataset of images to use for training the HardAssigner
     * @param clusters the number of clusters to assign vectors into
     * @return a HardAssigner class
     */
    public HardAssigner<float[], float[], IntFloatPair> trainQuantiser(GroupedDataset<String, VFSListDataset<FImage>, FImage> images, int clusters){
        List<FloatFV> vectors = new ArrayList<>();

        //collect feature vectors from images
        System.out.println("collecting feature vectors...");
        for(ListDataset<FImage> list : images.values()) {
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

    /**
     * A custom feature extractor class which uses bag of words
     */
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

    /**
     * Trains the classifier on a set of images
     * @param training images to train the classifier on
     */
    public void train(GroupedDataset<String, VFSListDataset<FImage>, FImage> training){

        //train a quantiser using a random sample of n images from the training dataset
        HardAssigner<float[], float[], IntFloatPair> assigner = trainQuantiser(training, 500);

        FeatureExtractor<DoubleFV, FImage> extractor = new Extractor(assigner);

        classifier = new LiblinearAnnotator<>(
                extractor, LiblinearAnnotator.Mode.MULTILABEL, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);

        System.out.println("training...");
//        stratified sampling used to speed up for development
//        classifier.train(new StratifiedGroupedUniformRandomisedSampler(2).sample(training));
        classifier.train(training);
        System.out.println("training complete");
    }

    /**
     * Applies the trained annotator to a set of labelled images and classifies them
     * @param testing dataset of images to classify
     */
    public void classifyOnTrainingData(GroupedDataset<String, ListDataset<FImage>, FImage> testing){
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

    /**
     * Applies the trained annotator to a set of images and classifies them
     * @param testing dataset of images to classify
     */
    public void classify(VFSListDataset<FImage> testing) throws IOException {
        int counter = 0;
        ArrayList<String> results = new ArrayList<>();
        BufferedWriter writer = new BufferedWriter(new FileWriter("Run2.txt"));
        for (FImage i:testing){
            //(uncomment the following line and comment out the one after if you want to view all the classes with their confidence rates for each image)
            //System.out.println("image"+counter+".jpg" + " " + getHighestConfidentClass(i) + " ---- " + getClassConfidence(i));
            ClassificationResult result = classifier.classify(i);
            results.add(testing.getID(counter).split("/")[1] + " " + HighestConfidence.getHighestConfidenceClass(result));
            counter++;
        }
        //The results are not in ascending order of image name, so these are sorted along with their classifications into the correct order
        Collections.sort(results, new Comparator<String>() {
            @Override
            public int compare(String o1, String o2) {
                int comparingInt = Integer.compare(Integer.parseInt(o1.split(" ")[0].split("\\.")[0]), Integer.parseInt(o2.split(" ")[0].split("\\.")[0]));
                if (comparingInt != 0) {
                    return comparingInt;
                }
                return o1.compareTo(o2);
            }
        });

        //The sorted answers are now saved to the text file
        for (String x:results){
            writer.write(x);
            writer.newLine();
        }
        writer.close();
    }

    /**
     * Converts a list of feature vectors to a 2D array
     * @param list list of feature vectors
     * @return 2D array
     */
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
