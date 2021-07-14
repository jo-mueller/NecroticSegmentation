import qupath.lib.images.servers.LabeledImageServer

def imageData = getCurrentImageData()

// Define output params
double requestedPixelSize = 2.5;
int requestedTileSize = 256;

// Define output path (relative to project)
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
def pathOutput = buildFilePath(PROJECT_BASE_DIR, 'tiles_' + String.valueOf(requestedTileSize) + '_' + String.valueOf(requestedPixelSize))
mkdirs(pathOutput)

// Convert to downsample
double downsample = requestedPixelSize / imageData.getServer().getPixelCalibration().getAveragedPixelSize()

// Create an ImageServer where the pixels are derived from annotations
def labelServer = new LabeledImageServer.Builder(imageData)
    .backgroundLabel(0, ColorTools.BLACK) // Specify background label (usually 0 or 255)
    .downsample(downsample)    // Choose server resolution; this should match the resolution at which tiles are exported
//    .addLabel('EmptyTiles', 1)      // Choose output labels (the order matters!)
    .addLabel('NonVital', 1)
    .addLabel('Vital', 2)
    .addLabel('SmoothMuscle', 3)
    .addLabel('CutArtifact', 4)
    .multichannelOutput(true)  // If true, each label is a different channel (required for multiclass probability)
    .build()

// Create an exporter that requests corresponding tiles from the original & labeled image servers
selectAnnotations()
new TileExporter(imageData)
    .downsample(downsample)     // Define export resolution
    .imageExtension('.tif')     // Define file extension for original pixels (often .tif, .jpg, '.png' or '.ome.tif')
    .tileSize(requestedTileSize)// Define size of each tile, in pixels
    .labeledServer(labelServer) // Define the labeled image server to use (i.e. the one we just built)
    .annotatedTilesOnly(true)  // If true, only export tiles if there is a (labeled) annotation present
    .overlap(64)                // Define overlap, in pixel units at the export resolution
    .writeTiles(pathOutput)     // Write tiles to the specified directory

print 'Done!'