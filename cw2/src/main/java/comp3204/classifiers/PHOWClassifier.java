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

    // Trains the classifier
    public void train(GroupedDataset<String, ListDataset<FImage>, FImage> training){
        DenseSIFT dsift = new DenseSIFT(5, 7);
        final PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<FImage>(dsift, 6f, 7);

        final HardAssigner<byte[], float[], IntFloatPair> assigner =
                getAssigner(GroupedUniformRandomisedSampler.sample(training, 30), pdsift);

        FeatureExtractor<DoubleFV, FImage> extractor = new FeatureExtractor<DoubleFV, FImage>() {
            @Override
            public DoubleFV extractFeature(FImage object) {
                FImage image = object.getImage();
                pdsift.analyseImage(image);

                BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<byte[]>(assigner);

                BlockSpatialAggregator<byte[], SparseIntFV> spatial = new BlockSpatialAggregator<byte[], SparseIntFV>(
                        bovw, 2, 2);

                return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
            }
        };

        ann = new LiblinearAnnotator<>(extractor, LiblinearAnnotator.Mode.MULTICLASS,
                SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);

        ann.train(training);
    }

    public void classify(GroupedDataset<String, ListDataset<FImage>, FImage> testing) {
        int count = 0;
        for (FImage i: testing) {
            System.out.println(count + ".jpg" + " " + ann.classify(i).getPredictedClasses().toString().replace("[", "").replace("]", ""));
            count++;
        }
    }

    // returns a HardAssigner used to assign SIFT features to identifiers
    public HardAssigner<byte[],float[],IntFloatPair> getAssigner(GroupedDataset<String, ListDataset<FImage>, FImage> training, PyramidDenseSIFT pdsift){
        // Stores list of keypoint descriptors
        List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();
        int count = 0;
        for (FImage rec : training) {
            System.out.println("Count " + count);
            FImage img = rec.getImage();

            // Analyses each image and extracts DSIFT descriptors
            pdsift.analyseImage(img);
            allkeys.add(pdsift.getByteKeypoints(0.005f));
            count++;
        }

        // takes the first 10000 features and clusters them into 300 classes
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
