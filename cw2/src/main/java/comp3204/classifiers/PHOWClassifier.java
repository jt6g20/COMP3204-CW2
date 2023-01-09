package comp3204.classifiers;

import comp3204.utility.HighestConfidence;
import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
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
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.util.pair.IntFloatPair;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.*;

public class PHOWClassifier {

    LiblinearAnnotator<FImage, String> classifier;

    /**
     * Trains a linear classifier by extracting DenseSIFT features using a PHOW technique
     * @param training The training dataset
     */
    public void train(GroupedDataset<String, VFSListDataset<FImage>, FImage> training){
        // Constructs DenseSIFT extractor object, with a step size of 5 px window size of 7 px
        DenseSIFT dsift = new DenseSIFT(5, 7);
        // Constructs a Pyramid DenseSIFT extractor using the object above, with a magnification factor of 6 (controls smoothing)
        final PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<>(dsift, 6f, 7);

        // Instantiates and trains a HardAssigner using a random sample of 30 images across all groups from training data
        final HardAssigner<byte[], float[], IntFloatPair> assigner =
                getAssigner(GroupedUniformRandomisedSampler.sample(training, 30), pdsift);

        // Takes in the Pyramid DenseSIFT to construct a PHOW extractor
        FeatureExtractor<DoubleFV, FImage> phowExtractor = new FeatureExtractor<DoubleFV, FImage>() {
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

        // Wraps phowExtractor in a HomogeneousKernelMap
        HomogeneousKernelMap kernelMap = new HomogeneousKernelMap(HomogeneousKernelMap.KernelType.Chi2, HomogeneousKernelMap.WindowType.Rectangular);
        FeatureExtractor<DoubleFV, FImage> featureExtractor = kernelMap.createWrappedExtractor(phowExtractor);

        // Instantiates a Linear classifier and trains it using our extractor
        classifier = new LiblinearAnnotator<>(featureExtractor, LiblinearAnnotator.Mode.MULTICLASS,
                SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
        classifier.train(training);

    }

    /**
     * Applied the trained annotator to a set of images and classifies them
     * @param testing dataset
     */
    public void classify(VFSListDataset<FImage> testing) throws Exception {
        int counter = 0;
        ArrayList<String> results = new ArrayList<>();
        BufferedWriter writer = new BufferedWriter(new FileWriter("Run3.txt"));
        for (FImage i:testing) {
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
     * Constructs a HardAssigner trained on the dataset to perform K-means clustering on SIFT features
     * @param training dataset of FImages
     * @param pdsift Pyramid DenseSIFT
     * @return a HardAssigner Class
     */
    public HardAssigner<byte[],float[],IntFloatPair> getAssigner(GroupedDataset<String, ListDataset<FImage>, FImage> training, PyramidDenseSIFT<FImage> pdsift){
        // Stores list of keypoint descriptors
        List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<>();

        // Iterates through the training
        for (FImage rec : training) {
            FImage img = rec.getImage();

            // Analyses each image and extracts DSIFT descriptors
            pdsift.analyseImage(img);
            allkeys.add(pdsift.getByteKeypoints(0.005f));
        }

        // Takes the first 10000 features and clusters them into 300 classes
        if (allkeys.size() > 10000)
            allkeys = allkeys.subList(0, 10000);

        ByteKMeans km = ByteKMeans.createKDTreeEnsemble(300);
        DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allkeys);
        ByteCentroidsResult result = km.cluster(datasource);

        return result.defaultHardAssigner();
    }

    public LiblinearAnnotator<FImage, String> getClassifier() {
        return classifier;
    }
}
