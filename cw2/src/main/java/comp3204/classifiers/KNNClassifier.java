package comp3204.classifiers;

import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.ml.annotation.basic.KNNAnnotator;

public class KNNClassifier {

    public FImage imageResize(FImage image){
        //Crops the image about the centre into a square shape with a 1:1 aspect ratio
        int imageCrop = Math.min(image.width, image.height);
        FImage croppedImage = image.extractCenter(imageCrop,imageCrop);
        //Resizes this image down to a 16x16 image
        return croppedImage.processInplace(new ResizeProcessor(16,16));
    }

    //Packs all the pixel values into a single vector by concatenating the image rows
    public float[] concatImgRowsToVec(FImage image){
        return image.getPixelVectorNative(new float[image.getWidth() * image.getHeight()]);
    }

    /*public void train(VFSListDataset<FImage> training){
        // Create a KNN classifier
        KNNAnnotator<FImage, String, FImage> knn = KNNAnnotator.create(training, 1);

        // Classify the test images
        for (FImage image : training) {
            String predictedClass = knn.classify(image).getAnnotations().iterator().next();
            System.out.println(image.getName() + " " + predictedClass);
        }
    }*/
}
