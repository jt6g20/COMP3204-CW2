package comp3204.classifiers;

import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.feature.FloatFV;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;

import java.util.ArrayList;
import java.util.Iterator;

public class OVAClassifier {
    LiblinearAnnotator ova;

    public float[][] extractPatchVectors(FImage image){

        int patchSize = 8;
        int increment = 4;

        ArrayList<float[]> vectors = new ArrayList<>();

        int xIncrements = (int) Math.floor(image.getWidth() / increment);
        int yIncrements = (int) Math.floor(image.getHeight() / increment);

        for (int y = 0; y < yIncrements; y++) {
            for (int x = 0; x < xIncrements; x++) {
                FImage sample = image.extractROI(x*increment,y*increment, patchSize, patchSize);
                vectors.add(sample.getPixelVectorNative(new float[sample.getWidth() * sample.getHeight()]));
            }
        }

        return vectors.toArray(new float[vectors.get(0).length][vectors.size()]);
    }

    public void train(GroupedDataset<String, ListDataset<FImage>, FImage> training){
//        ova = LiblinearAnnotator(extractor, LiblinearAnnotator.Mode.MULTILABEL);
    }
}
