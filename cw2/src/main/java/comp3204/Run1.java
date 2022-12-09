package comp3204;

import comp3204.utility.Data;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;

import java.net.URISyntaxException;

public class Run1 {
    public static void main( String[] args ) throws FileSystemException, URISyntaxException {
        
        GroupedDataset<String, VFSListDataset<FImage>, FImage> training = Data.training();
        System.out.println(training.size());

        VFSListDataset<FImage> testing = Data.testing();
        System.out.println(testing.size());
    }
}
