package comp3204.classifiers;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;

public class OVAClassifier {
    LiblinearAnnotator ova;

    public void train(GroupedDataset<String, ListDataset<FImage>, FImage> training){
//        ova = LiblinearAnnotator(extractor, LiblinearAnnotator.Mode.MULTILABEL);
    }
}
