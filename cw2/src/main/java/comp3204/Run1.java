package comp3204;

import comp3204.classifiers.KNNClassifier;
import comp3204.utility.Data;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;

import java.net.URISyntaxException;
import java.util.*;

public class Run1 {
    public static void main( String[] args ) throws FileSystemException, URISyntaxException {
        
        GroupedDataset<String, VFSListDataset<FImage>, FImage> training = Data.training();
        System.out.println("Training set size: " + training.size());
        System.out.println("Training set classes:" + training.getGroups());
        System.out.println("______________________________________________________________");
        for (String i: training.getGroups()){
            System.out.println("Training class name: " + i + " ||||| class info:" + training.get(i));
        }

        /*VFSListDataset<FImage> testing = Data.testing();
        System.out.println("Testing set size: " + testing.size());


        // Testing knn image cropping
        KNNClassifier knn = new KNNClassifier();
        FImage croppedImage = knn.imageResize(testing.get(0));
        //DisplayUtilities.display(testing.get(0));
        DisplayUtilities.display(croppedImage);
        float[] packedImgPixelVec = knn.concatImgRowsToVec(croppedImage);

        //The following 4 lines can be safely deleted once they are no longer needed
        System.out.println("Image breakdown:");
        showMatrix(croppedImage.pixels); //Prints 16x16 matrix of the cropped image for diagnosing purposes
        System.out.println("________________________________________________");
        System.out.println(Arrays.toString(packedImgPixelVec)); //Prints out packed img vector*/

    }

    //Prints out a 2D vector in the form of a matrix
    public static void showMatrix(float[][] mat) {
        for (float[] row: mat)
            System.out.println(Arrays.toString(row));
    }

}
