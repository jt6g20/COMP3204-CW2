package comp3204.utility;

import comp3204.Run1;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

import java.io.File;
import java.net.URISyntaxException;

public class Data {
    /**
     * utility function for fetching and formatting training dataset
     * @return OpenIMAJ-compatible training dataset
     * @throws FileSystemException
     * @throws URISyntaxException
     */
    public static VFSGroupDataset<FImage> training() throws FileSystemException, URISyntaxException{
        File file = new File(Run1.class.getClassLoader().getResource("comp3204/training.zip").toURI());
        String path = file.getAbsolutePath();
        VFSGroupDataset training = new VFSGroupDataset<>("zip:" + path, ImageUtilities.FIMAGE_READER);
        training.remove(training.keySet().toArray()[0]);
        return training;
    }

    /**
     * utility function for fetching and formatting testing dataset
     * @return OpenIMAJ-compatible testing dataset
     * @throws URISyntaxException
     * @throws FileSystemException
     */
    public static VFSListDataset<FImage> testing() throws URISyntaxException, FileSystemException {
        File file = new File(Run1.class.getClassLoader().getResource("comp3204/testing.zip").toURI());
        String path = file.getAbsolutePath();
        return new VFSListDataset<>("zip:" + path, ImageUtilities.FIMAGE_READER);
    }
}
