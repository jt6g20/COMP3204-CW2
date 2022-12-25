package comp3204;

import comp3204.classifiers.KNNClassifier;
import comp3204.utility.Data;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;

import java.net.URISyntaxException;
import java.util.Arrays;

public class Run1 {
    public static void main( String[] args ) throws FileSystemException, URISyntaxException {
        
        GroupedDataset<String, VFSListDataset<FImage>, FImage> training = Data.training();
        System.out.println(training.size());

        VFSListDataset<FImage> testing = Data.testing();
        System.out.println(testing.size());

        // Testing knn image cropping
        KNNClassifier knn = new KNNClassifier();
        FImage croppedImage = knn.imageResize(testing.get(0));
        //DisplayUtilities.display(testing.get(0));
        DisplayUtilities.display(croppedImage);

        System.out.println("Pixel Diagnostics:");
        showMatrix(croppedImage.pixels);
        System.out.println("________________________________________________");
        System.out.println(Arrays.toString(croppedImage.getPixelVectorNative(new float[croppedImage.getWidth() * croppedImage.getHeight()])));

    }

    //Prints out a 2D vector in the form of a matrix
    public static void showMatrix(float[][] mat) {
        for (float[] row: mat)
            System.out.println(Arrays.toString(row));
    }

}
