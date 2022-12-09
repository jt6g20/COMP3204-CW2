package comp3204.classifiers;

import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;

public class KNNClassifier {

    public FImage imageResize(FImage image){
        int imageCrop = Math.min(image.width, image.height);
        FImage croppedImage = image.extractCenter(imageCrop,imageCrop);
        return croppedImage.processInplace(new ResizeProcessor(16,16));
    }
}
