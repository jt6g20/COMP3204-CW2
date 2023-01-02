package comp3204;

import comp3204.utility.Data;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

import java.io.File;
import java.net.URISyntaxException;

public class Run2 {
    public static void main( String[] args ) throws FileSystemException, URISyntaxException {
        GroupedDataset<String, VFSListDataset<FImage>, FImage> training = Data.training();

        System.out.println("Training set size: " + training.size());
        System.out.println("Training set classes:" + training.getGroups());

    }
}
