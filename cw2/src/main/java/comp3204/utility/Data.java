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
    public static VFSGroupDataset<FImage> training() throws FileSystemException, URISyntaxException{
        File file = new File(Run1.class.getClassLoader().getResource("comp3204/training.zip").toURI());
        System.out.println(file);
        String path = file.getAbsolutePath();
        System.out.println(path);

        VFSGroupDataset onedere = new VFSGroupDataset<>("zip:" + path, ImageUtilities.FIMAGE_READER);
        System.out.println("KEY SET" + onedere.keySet());
        System.out.println(onedere.remove(onedere.keySet().toArray()[0]));

        System.out.println(onedere.getGroups());
        return new VFSGroupDataset<>("zip:" + path, ImageUtilities.FIMAGE_READER);
    }

    public static VFSListDataset<FImage> testing() throws URISyntaxException, FileSystemException {
        File file = new File(Run1.class.getClassLoader().getResource("comp3204/testing.zip").toURI());
        String path = file.getAbsolutePath();
        return new VFSListDataset<>("zip:" + path, ImageUtilities.FIMAGE_READER);
    }
}
