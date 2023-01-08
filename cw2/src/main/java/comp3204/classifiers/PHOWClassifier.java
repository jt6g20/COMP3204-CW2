package comp3204.classifiers;

import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.util.pair.IntFloatPair;

import java.util.*;

public class PHOWClassifier {

    LiblinearAnnotator<FImage, String> ann;

    /**
     * Trains a linear classifier by extracting DenseSIFT features using a PHOW technique
     * @param training The training dataset
     */
    public void train(GroupedDataset<String, ListDataset<FImage>, FImage> training){
        // Constructs DenseSIFT extractor object, with a step size of 5 px window size of 7 px
        DenseSIFT dsift = new DenseSIFT(5, 7);
        // Constructs a Pyramid DenseSIFT extractor using the object above, with a magnification factor of 6 (controls smoothing)
        final PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<>(dsift, 6f, 7);

        // Instantiates and trains a HardAssigner using a random sample of 30 images across all groups from training data
        final HardAssigner<byte[], float[], IntFloatPair> assigner =
                getAssigner(GroupedUniformRandomisedSampler.sample(training, 30), pdsift);

        // Takes in the Pyramid DenseSIFT to construct a PHOW extractor
        FeatureExtractor<DoubleFV, FImage> extractor = new FeatureExtractor<DoubleFV, FImage>() {
            /**
             * Extracts DenseSIFT features from an image
             * @param object the FImage to extract from
             * @return A spatial histogram representing the visual word occurrences, in vector form
             */
            @Override
            public DoubleFV extractFeature(FImage object) {
                FImage image = object.getImage();
                pdsift.analyseImage(image);

                // Uses the HardAssigner to assign each feature to a visual word
                BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<>(assigner);

                // Breaks the image into 4 blocks and computes the histogram for each one
                BlockSpatialAggregator<byte[], SparseIntFV> spatial = new BlockSpatialAggregator<>(
                        bovw, 2, 2);

                // returns the resultant spatial histograms, appended and normalised
                return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
            }
        };

        // Instantiates a Linear classifier and trains it using our PHOW extractor
        ann = new LiblinearAnnotator<>(extractor, LiblinearAnnotator.Mode.MULTICLASS,
                SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
        ann.train(training);
    }

    /**
     * Applied the trained annotator to a set of images and classifies them
     * @param testing dataset
     */
    public void classify(GroupedDataset<String, ListDataset<FImage>, FImage> testing) {
        int count = 0;
        for (FImage i: testing) {
            System.out.println(count + ".jpg" + " " + ann.classify(i).getPredictedClasses().toString().replace("[", "").replace("]", ""));
            count++;
        }
    }

    /**
     * Constructs a HardAssigner trained on the dataset to perform K-means clustering on SIFT features
     * @param training dataset of FImages
     * @param pdsift Pyramid DenseSIFT
     * @return a HardAssigner Class
     */
    public HardAssigner<byte[],float[],IntFloatPair> getAssigner(GroupedDataset<String, ListDataset<FImage>, FImage> training, PyramidDenseSIFT<FImage> pdsift){
        // Stores list of keypoint descriptors
        List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<>();

        // Iterates through the training
        int count = 0;
        for (FImage rec : training) {
            System.out.println("Count " + count);
            FImage img = rec.getImage();

            // Analyses each image and extracts DSIFT descriptors
            pdsift.analyseImage(img);
            allkeys.add(pdsift.getByteKeypoints(0.005f));
            count++;
        }

        // Takes the first 10000 features and clusters them into 300 classes
        if (allkeys.size() > 10000)
            allkeys = allkeys.subList(0, 10000);

        ByteKMeans km = ByteKMeans.createKDTreeEnsemble(300);
        DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allkeys);
        ByteCentroidsResult result = km.cluster(datasource);

        return result.defaultHardAssigner();
    }

    public LiblinearAnnotator<FImage, String> getAnn() {
        return ann;
    }
}
