/*
 * Converts an HE image and the respective label-map into
 * patches of defined size
 */

close("*");

// Input
#@ File (label="Selected sample file", style="file") fname_HE_image
#@ File (label="Output directory", style="directory") output_directory

root =  File.getParent(fname_HE_image) + "/";
output_directory = output_directory + "/";

// Config
run("CLIJ2 Macro Extensions", "cl_device=[GeForce RTX 3060 Ti]");

smoothing_vital = 3;
smoothing_mask = 6;

// find and open labelmap
fname_HE_label = root + "3_meas/HE_seg_Simple_Segmentation.tif";
open(fname_HE_label);
HE_label = getTitle();

// open BlackTile ROI
roiManager("open", root + "1_seg/BlackTiles.zip");
selectWindow(HE_label);
roiManager("select", 0);
run("Set...", "value=3");
roiManager("reset");

// open raw
run("Bio-Formats Importer", "open=" + fname_HE_image + " autoscale color_mode=Composite series_3");
rename(File.getName(fname_HE_image));
HE_raw = getTitle();

// Process labelmap
HE_label = MakeLabelMap(HE_label, smoothing_vital, smoothing_mask);
HE_label = OverlayCorrect(HE_label, HE_raw);
Patchify(HE_raw, HE_label, 512, output_directory);



function Patchify(image, labels, size, outdir){

	selectWindow(labels);
	setLocation(0, 0);

	selectWindow(image);
	setLocation(screenWidth/2, 0);

	if (!File.exists(outdir + "Tiles/")) {
		File.makeDirectory(outdir + "Tiles/");
	}
	
	selectWindow(image);
	width = getWidth();
	height = getHeight();

	Nx = floor(width/size);
	Ny = floor(height/size);

	// First iteration
	for (ix = 0; ix < Nx; ix++) {
		for (iy = 0; iy < Ny; iy++) {

			// for every tile, check amount of background. 
			// If too much, skip.
			selectWindow(labels);
			setThreshold(0, 0);
			run("Measure");	

			// Patchify Labels
			makeRectangle(ix*size, iy*size, size, size);
			if (getResult("%Area", nResults-1) > 50) {
				continue;
			}	

			selectWindow(labels);
			Tile2File(labels, image + "_ix" + ix + "_iy" + iy + "_labels", outdir + "Tiles/");
			
			// Patchify image
			selectWindow(image);	
			makeRectangle(ix*size, iy*size, size, size);
			Tile2File(image, image + "_ix" + ix + "_iy" + iy + "_image", outdir + "Tiles/");
		}
	}
	// Second iteration
	for (ix = 0; ix < Nx; ix++) {
		for (iy = 0; iy < Ny; iy++) {

			// for every tile, check amount of background. 
			// If too much, skip.
			selectWindow(labels);
			setThreshold(0, 0);
			run("Measure");	

			// Only full tiles
			if (ix*size + 3*size/2 > getWidth() || iy*size + 3*size/2 > getHeight()) {
				continue;
			}
			// Patchify Labels
			makeRectangle(ix*size + floor(size/2), iy*size + floor(size/2), size, size);
			if (getResult("%Area", nResults-1) > 50) {
				continue;
			}	

			selectWindow(labels);
			Tile2File(labels, image + "_2ix" + ix + "_2iy" + iy + "_labels", outdir + "Tiles/");
			
			// Patchify image
			selectWindow(image);	
			makeRectangle(ix*size, iy*size, size, size);
			Tile2File(image, image + "_2ix" + ix + "_2iy" + iy + "_image", outdir + "Tiles/");
		}
	}
}

function Tile2File(image, tilename, dir){
	selectWindow(image);
	run("Duplicate...", "title=" + tilename + " stack");
	saveAs("tif", dir + tilename);
	close();
}

function OverlayCorrect(image, overlay){
	selectWindow(overlay);
	run("RGB Color");
	RGB = getTitle();

	selectWindow(image);
	run("RGB Color");

	run("Add Image...", "image=["+RGB+"] x=0 y=0 opacity=80");
	waitForUser("Contouring");
	run("Remove Overlay");
	close(RGB);	

	selectWindow(image);
	run("8-bit");
	return image;
}

function MakeLabelMap(image, n_vital, n_mask) {
	// function description

	selectWindow(image);
	run("Scale...", "x=0.25 y=0.25" +
					" width=" + getWidth() +
					" height=" + getHeight() +
					" interpolation=None"+
					" create");
	image_small = getTitle();
	close(image);

	selectWindow(image_small);
	Ext.CLIJ2_clear();

	// push and convert
	Ext.CLIJ2_push(image_small);
	Ext.CLIJ2_convertUInt8(image_small, uint8);
	
	// Make tumor mask
	Ext.CLIJ2_equalConstant(uint8, mask, 3);
	Ext.CLIJ2_openingBox(mask, Background, n_mask);
	Ext.CLIJ2_binaryNot(Background, tmp);
	Ext.CLIJx_imageJFillHoles(tmp, TumorMask);
	
	// Make vital mask
	Ext.CLIJ2_equalConstant(uint8, vital, 1);
	for (i = 1; i <= n_vital; i++) {

		// destructive smoothing
		Ext.CLIJ2_openingBox(vital, tmp, Math.pow(2, i));
		Ext.CLIJ2_closingBox(tmp, vital, Math.pow(2, i));
		
		/*
		 // connective smoothing
		Ext.CLIJ2_openingBox(mask, tmp, Math.pow(2, i));
		Ext.CLIJ2_closingBox(tmp, mask, Math.pow(2, i));
		*/
	}
	Ext.CLIJ2_binaryAnd(vital, TumorMask, tmp);
	Ext.CLIJ2_addImages(tmp, TumorMask, labelmap);

	// crop to limits of tumor mask
	Ext.CLIJ2_binaryNot(Background, tmp);
	Ext.CLIJ2_multiplyImages(labelmap, tmp, output);

	close(image_small);
	Ext.CLIJ2_pull(output);
	rename("Mask_Vital");

	return "Mask_Vital";
}


