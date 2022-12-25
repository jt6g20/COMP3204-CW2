package comp3204.classifiers;

import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;

public class KNNClassifier {

    public FImage imageResize(FImage image){
        //Crops the image about the centre into a square shape with a 1:1 aspect ratio
        int imageCrop = Math.min(image.width, image.height);
        FImage croppedImage = image.extractCenter(imageCrop,imageCrop);
        //Resizes this image down to a 16x16 image
        return croppedImage.processInplace(new ResizeProcessor(16,16));
    }
}
