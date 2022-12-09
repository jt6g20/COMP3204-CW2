package comp3204;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

import java.io.File;

/**
 * OpenIMAJ Hello world!
 *
 */
public class Run2 {
    public static void main( String[] args ) {
        
        try {
            File file = new File(Run2.class.getClassLoader().getResource("comp3204/training.zip").toURI());
            String path = file.getAbsolutePath();
            GroupedDataset<String, VFSListDataset<FImage>, FImage> training = new VFSGroupDataset<>("zip:" + path, ImageUtilities.FIMAGE_READER);
            System.out.println(training.size());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
