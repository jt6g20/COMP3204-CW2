package comp3204;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;

import java.io.File;

/**
 * OpenIMAJ Hello world!
 *
 */
public class App {
    public static void main( String[] args ) {
        
        try {
            File file = new File(App.class.getClassLoader().getResource("comp3204/training.zip").toURI());
            String path = file.getAbsolutePath();
            VFSListDataset<FImage> images = new VFSListDataset<>("zip:" + path, ImageUtilities.FIMAGE_READER);
            System.out.println(images.size());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
